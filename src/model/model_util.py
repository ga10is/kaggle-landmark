import os
import torch

from .. import config
from ..common.logger import get_logger


def save_checkpoint(state, is_best, fpath='checkpoint.pth'):
    torch.save(state, fpath)
    if is_best:
        torch.save(state, 'best_model.pth')


def load_checkpoint(_model, _metric_fc, _optimizer, _scheduler, fpath):
    checkpoint = torch.load(fpath)
    # reset optimizer setting
    _epoch = checkpoint['epoch']
    # TODO: back
    # _optimizer.load_state_dict(checkpoint['optimizer'])
    _scheduler.load_state_dict(checkpoint['scheduler'])
    _model.load_state_dict(checkpoint['state_dict'])
    # TODO: back
    # _metric_fc.load_state_dict(checkpoint['metric_fc'])

    return _epoch, _model, _metric_fc, _optimizer, _scheduler


def load_model(_model, _metric_fc, _optimizer, _scheduler):
    get_logger().info('start loading model: %s' % config.PRETRAIN_PATH)

    # create model object before following statement
    start_epoch, model, metric_fc, optimizer, scheduler = \
        load_checkpoint(_model, _metric_fc, _optimizer, _scheduler,
                        os.path.join(config.PRETRAIN_PATH, 'best_model.pth'))

    get_logger().info('end loading model')
    return start_epoch, model, metric_fc, optimizer, scheduler
