import os
import sys
from os.path import join

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = join(PATH, '../')
sys.path.append(ROOT)

from tensorboardX import SummaryWriter
import src.data_loader as data_loader
import datetime
import pytz
import logging
import src.powerboard as board
import src.utils as utils
import src.evals as evals
import src.model as model

MODEL = {
    'bgch': model.BGCH
}
LOSS_F = {
    'bgch': evals.bgch_loss
}

utils.set_seed(board.SEED)
print('--SEED--:', board.SEED)

dataset = data_loader.LoadData(data_name=board.args.dataset)
model = MODEL[board.args.model](dataset=dataset)
model = model.to(board.DEVICE)
loss_f = LOSS_F[board.args.model](model)

# log file path
path = join(board.BOARD_PATH, board.args.dataset)
timezone = pytz.timezone('Asia/Shanghai')
nowtime = datetime.datetime.now(tz=timezone)
log_path = join(path, nowtime.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + board.args.model)

record = 'bgch_result.txt'
grid_search_folder = os.path.join(board.BOARD_PATH, f"bgch/{board.args.dataset}")
if not os.path.exists(grid_search_folder):
    os.makedirs(grid_search_folder)
record_file = os.path.join(grid_search_folder, record)

# init tensorboard
if board.args.tensorboard:
    summarizer: SummaryWriter = SummaryWriter(log_path)
else:
    summarizer = None
    board.cprint('tensorboard disabled.')

try:
    max_recall20 = []
    # logger initializer
    log_name = utils.create_log_name(log_path)
    utils.log_config(path=log_path, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info('--------- model configuration ---------')
    for k in list(vars(board.args).keys()):
        logging.info('%s = %s' % (k, vars(board.args)[k]))

    gap = 5
    if board.args.dataset == 'dianping':
        gap = 1

    for epoch in range(board.args.epoch):
        info = evals.Train(dataset=dataset, model=model, epoch=epoch, loss_f=loss_f,
                           neg_ratio=board.args.neg_ratio, summarizer=summarizer)

        if epoch % gap == 0:
            board.cprint(f'--------- testing at epoch-{epoch} ---------')
            results = evals.Inference(dataset=dataset, model=model, epoch=epoch, summarizer=summarizer)

            logging.info(f'--------- testing at epoch-{epoch} ---------')
            logging.info(results)

        max_recall20.append(results['recall'][0])
        logging.info(f'EPOCH[{epoch + 1}/{board.args.epoch}] {info} ')

    with open(record_file, 'a') as write:
        write.write(nowtime.strftime("%m-%d-%Hh%Mm%Ss:  ") + str(max(max_recall20)) + "\n")
        logging.info(nowtime.strftime("%m-%d-%Hh%Mm%Ss:  ") + str(max(max_recall20)) + "\n")

except:
    with open(record_file, 'a') as write:
        write.write(nowtime.strftime("%m-%d-%Hh%Mm%Ss:  ") + "ERROR! \n")

    raise NotImplementedError('Error in running main file')


finally:
    if board.args.tensorboard:
        summarizer.close()
