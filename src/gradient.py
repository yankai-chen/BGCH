import torch
import torch.nn as nn
import torch.nn.functional as F
import src.powerboard as board
import src.data_loader as data_loader
import os
import math


class Gradient_Estimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


# **************************with rescaling**************************
class FS(Gradient_Estimator):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        binary_encoding = torch.sign(x)
        n = x[0].nelement()
        m = x.norm(1, 1, keepdim=True).div(n)
        scaler = m.expand(x.size())
        encodes = binary_encoding.mul(scaler)
        return encodes

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = torch.zeros_like(input)
        for i in range(1, board.FS_N, 2):
            output += 4.0 * board.FS_w / math.pi * torch.cos(i * board.FS_w * input)
        return torch.clamp(output, -1, 1) * grad_output
