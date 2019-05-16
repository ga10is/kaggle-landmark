import os
import torch

from .. import config
from ..common.logger import get_logger


def save_checkpoint(state, is_best, fpath='checkpoint.pth'):
    torch.save(state, fpath)
    if is_best:
        torch.save(state, 'best_model.pth')


def load_checkpoint(_model, _metric_fc, _optimizer, _scheduler, fpath, reset_epoch=False):
    checkpoint = torch.load(fpath)
    _epoch = 0
    if reset_epoch is False:
        # reset optimizer setting
        _epoch = checkpoint['epoch']
        _optimizer.load_state_dict(checkpoint['optimizer'])
        _scheduler.load_state_dict(checkpoint['scheduler'])
    _model.load_state_dict(checkpoint['state_dict'])
    _metric_fc.load_state_dict(checkpoint['metric_fc'])

    return _epoch, _model, _metric_fc, _optimizer, _scheduler


def load_model(_model, _metric_fc, _optimizer, _scheduler, reset_epoch=False):
    get_logger().info('start loading model: %s' % config.PRETRAIN_PATH)
    if reset_epoch is True:
        get_logger().info('reset optimizer setting. start epoch from 0')

    # create model object before following statement
    start_epoch, model, metric_fc, optimizer, scheduler = \
        load_checkpoint(_model, _metric_fc, _optimizer, _scheduler,
                        os.path.join(config.PRETRAIN_PATH, 'best_model.pth'),
                        reset_epoch=reset_epoch)

    get_logger().info('end loading model')
    return start_epoch, model, metric_fc, optimizer, scheduler
