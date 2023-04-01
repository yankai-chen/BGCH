import torch
import torch.nn as nn
import src.powerboard as board
import src.gradient as gradient
import src.data_loader as data_loader
import torch.nn.functional as F
import numpy as np


class Basic_Model(nn.Module):
    def __init__(self):
        super(Basic_Model, self).__init__()

    def get_scores(self, user_index):
        raise NotImplementedError


class Basic_Hash_Layer(nn.Module):
    def __init__(self):
        super(Basic_Hash_Layer, self).__init__()


class Hashing_Layer(Basic_Hash_Layer):
    def __init__(self):
        super(Hashing_Layer, self).__init__()
        self.con_dim = board.args.con_dim
        self.bin_dim = board.args.bin_dim
        self.fc_num = board.args.fc_num

        self.FC_buckets = []
        mlp = nn.Sequential()
        if self.fc_num == 1:
            mlp.add_module(name='fc-layer-{}'.format(0),
                                module=nn.Linear(in_features=self.con_dim, out_features=self.bin_dim, bias=False))
        else:
            for id in range(self.fc_num - 1):
                mlp.add_module(name='fc-layer-{}'.format(id),
                                    module=nn.Linear(in_features=self.con_dim, out_features=self.con_dim, bias=False))

            mlp.add_module(name='fc-layer-{}'.format(self.fc_num),
                                module=nn.Linear(in_features=self.con_dim, out_features=self.bin_dim, bias=False))
        self.FC_buckets.append(mlp.to(board.DEVICE))

    def binarize(self, X):
        encode = gradient.FS.apply(X)
        return encode

    def forward(self, X):
        output = self.FC_buckets[0](X)
        hash_codes = self.binarize(output)
        return hash_codes


class BGCH(Basic_Model):
    def __init__(self, dataset):
        super(BGCH, self).__init__()
        self.dataset: data_loader.LoadData = dataset
        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.con_dim = board.args.con_dim
        self.bin_dim = board.args.bin_dim
        self.num_layers = board.args.layers
        self.eps = board.args.eps
        self.f = nn.Sigmoid()

        self.with_binarize = False

        self.user_cont_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.con_dim).to(board.DEVICE)
        self.item_cont_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.con_dim).to(board.DEVICE)

        nn.init.normal_(self.user_cont_embed.weight, std=0.1)
        nn.init.normal_(self.item_cont_embed.weight, std=0.1)
        board.cprint('initializing with NORMAL distribution.')

        self.G = self.dataset.load_sparse_graph()
        self.hashing_layer = Hashing_Layer()

    def _binarize(self, embedding):
        return self.hashing_layer(embedding)


    def random_projection(self, X, num_dim=1, variance=board.args.RP_variance,
                          iteration=board.args.RP_iteration, eta=board.args.RP_eta, avg=board.args.RP_avg):
        v = torch.zeros(X.shape[1], num_dim, dtype=torch.float32).to(board.DEVICE)
        for _ in range(avg):
            vk = (variance ** 0.5) * torch.randn(X.shape[1], num_dim).to(board.DEVICE)
            for _ in range(iteration):
                vk = torch.mm(torch.mm(X.t(), X), vk)
            v += vk
        v /= avg
        Xk = X - eta * (torch.mm(torch.mm(X, v), v.t())) / v.pow(2).sum()
        return Xk

    def aggregate_embed(self):
        if self.with_binarize:
            user_embed = self.random_projection(self.user_cont_embed.weight)
            item_embed = self.random_projection(self.item_cont_embed.weight)
        else:
            user_embed = self.random_projection(self.user_cont_embed.weight)
            item_embed = self.random_projection(self.item_cont_embed.weight)

        all_embed = torch.cat([user_embed, item_embed])
        embed_list = [self._binarize(all_embed)]
        g = self.G

        for id in range(self.num_layers):
            all_embed = torch.sparse.mm(g, all_embed)
            embed_list.append(self._binarize(all_embed))

        embed_list = torch.cat(embed_list, dim=1)

        users_embed, items_embed = torch.split(embed_list, [self.num_users, self.num_items])
        return users_embed, items_embed

    def _recons_loss(self, user_con_embed, pos_con_embed, neg_con_embed):
        pos_norm_score = self.f(torch.sum(user_con_embed * pos_con_embed, dim=-1))
        neg_norm_score = self.f(torch.sum(user_con_embed * neg_con_embed, dim=-1))
        labels0 = torch.zeros_like(neg_norm_score, dtype=torch.float32)
        labels1 = torch.ones_like(pos_norm_score, dtype=torch.float32)
        scores = torch.cat([pos_norm_score, neg_norm_score], dim=0)
        labels = torch.cat([labels1, labels0], dim=0)
        loss = torch.mean(nn.BCELoss()(scores, labels))
        return loss

    def _BPR_loss(self, user_con_embed, pos_con_embed, neg_con_embed):
        pos_scores = torch.mul(user_con_embed, pos_con_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_con_embed, neg_con_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def loss(self, user_index, pos_index, neg_index):
        user_con_embed = self.user_cont_embed(user_index)
        pos_con_embed = self.item_cont_embed(pos_index)
        neg_con_embed = self.item_cont_embed(neg_index)

        all_agg_user_embed, all_agg_item_embed = self.aggregate_embed()

        user_agg_embed = all_agg_user_embed[user_index]
        pos_agg_embed = all_agg_item_embed[pos_index]
        neg_agg_embed = all_agg_item_embed[neg_index]

        reg_loss = (1 / 2) * (self.user_cont_embed.weight.norm(2).pow(2) +
                              self.item_cont_embed.weight.norm(2).pow(2)) / float(len(user_index))

        reg_loss += (1 / 2) * (all_agg_user_embed.norm(2).pow(2) +
                               all_agg_item_embed.norm(2).pow(2)) / float(len(user_index))

        for bucket in self.hashing_layer.FC_buckets:
            for linear in bucket:
                reg_loss += (1 / 2) * (linear.weight.norm(2).pow(2)) / float(len(user_index))

        loss1 = self._BPR_loss(user_agg_embed, pos_agg_embed, neg_agg_embed)
        loss2 = self._recons_loss(user_con_embed, pos_con_embed, neg_con_embed)
        return loss1, loss2, reg_loss

    def get_scores(self, user_index):
        all_agg_user_embed, all_agg_item_embed = self.aggregate_embed()
        user_agg_embed = all_agg_user_embed[user_index]

        scores = torch.matmul(user_agg_embed, all_agg_item_embed.t())
        return scores




