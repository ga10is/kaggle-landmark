import pandas as pd
import joblib
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from . import config
from . import tvp
from .common.logger import create_logger, get_logger
from .preprocessing import get_exist_image, init_le
from .common.util import split_train_valid, debug_trace
from .dataset import LandmarkDataset
from .model.model import trn_trnsfms, tst_trnsfms, ResNet, save_checkpoint


@debug_trace
def init_model(le):
    print('model')
    n_classes = len(le.classes_)
    # Model
    # model = DenseNet(output_neurons=latent_dim, n_classes=len(le.classes_),  dropout_rate=0.5).cuda()
    _model = ResNet(output_neurons=config.latent_dim,
                    n_classes=n_classes, dropout_rate=0.5).cuda()

    print('metric_fc')
    # Last Layer
    # metric_fc = ArcMarginProduct(latent_dim, n_classes, s=60, m=0.5, easy_margin=False).cuda()
    _metric_fc = nn.Linear(config.latent_dim, n_classes).cuda()

    print('criterion')
    # Loss function
    _criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=2)

    print('optimizer')
    # Optimizer
    _optimizer = optim.Adam([{'params': _model.parameters()}, {
        'params': _metric_fc.parameters()}], lr=1e-4)
    # optimizer = optim.SGD([{'params':model.parameters()}, {'params': metric_fc.parameters()}],
    #                       lr=1e-3, momentum=0.9, weight_decay=1e-4)

    print('scheduler')
    # Scheduler
    mile_stones = [5, 7, 9, 10, 11, 12]
    _scheduler = optim.lr_scheduler.MultiStepLR(
        _optimizer, mile_stones, gamma=0.5, last_epoch=-1)
    # scheduler_step = EPOCHS
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, eta_min=1e-4)

    return _model, _metric_fc, _criterion, _optimizer, _scheduler


def train_main():
    get_logger().info('batch size: %d' % config.BATCH_SIZE_TRAIN)
    get_logger().info('epochs: %d' % config.EPOCHS)
    get_logger().info('latent_dim: %d' % config.latent_dim)

    # Train
    # load train data
    df_train = pd.read_csv(config.TRAIN_PATH, dtype={'id': 'object'})
    df_train.head()

    # create label encoder
    # must be init_le() before get_exist_image()
    label_encoder = init_le(df_train)
    joblib.dump(label_encoder, 'le.pkl')

    # use landmark_id which have more than 1 image
    df_train = get_exist_image(df_train, config.TRAIN_IMG_PATH)
    id_count = df_train['landmark_id'].value_counts()
    s_count = df_train['landmark_id'].map(id_count)
    df_train = df_train[s_count > 1]
    print('more than 1 landmark_id: %d, images: %d' %
          ((id_count > 1).sum(), df_train.shape[0]))

    # train validate split
    df_trn, df_val = split_train_valid(
        df_train, label_encoder.transform(df_train['landmark_id'].values), 10)
    get_logger().info('train num: %d, valid num: %d' % (len(df_trn), len(df_val)))

    # initialize Dataset and DataLoader
    train_dataset = LandmarkDataset(config.TRAIN_IMG_PATH, df_trn,
                                    trn_trnsfms, is_train=True, le=label_encoder)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True
                              )
    valid_dataset = LandmarkDataset(config.TRAIN_IMG_PATH, df_val,
                                    tst_trnsfms, is_train=True, le=label_encoder)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=False,
                              shuffle=False
                              )

    model, metric_fc, criterion, optimizer, scheduler \
        = init_model(label_encoder)

    get_logger().info('start epoch')
    start_epoch = 0
    best_score = 0
    for epoch in range(start_epoch + 1, config.EPOCHS + 1):
        scheduler.step()

        tvp.train(epoch, model, train_loader,
                  metric_fc, criterion, optimizer)

        valid_score = tvp.validate_arcface(model, metric_fc, valid_loader)

        is_best = valid_score > best_score
        print('best score (%f) at epoch (%d)' % (valid_score, epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'metric_fc': metric_fc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best)

    get_logger().info('end epoch')


if __name__ == '__main__':
    create_logger('landmark.log')

    train_main()
