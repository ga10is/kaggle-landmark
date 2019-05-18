import os
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from . import config
from . import tvp
from .common.logger import create_logger, get_logger
from .preprocessing import get_exist_image, init_le, select_train_data, split_train_valid_v2
from .common.util import debug_trace
from .dataset import LandmarkDataset
from .model.model import trn_trnsfms, tst_trnsfms, ResNet, DenseNet, DelfResNet
from .model.model_util import save_checkpoint, load_model
from .model.loss import ArcMarginProduct, FocalLoss, DummyLayer
from .delf.delf import Delf_V1


@debug_trace
def init_model(le):
    torch.backends.cudnn.benchmark = True

    print('model')
    n_classes = len(le.classes_)
    # Model
    # _model = DenseNet(output_neurons=config.latent_dim,
    #                 n_classes=n_classes, dropout_rate=config.DROPOUT_RATE).cuda()
    # _model = ResNet(output_neurons=config.latent_dim,
    #                 n_classes=n_classes, dropout_rate=config.DROPOUT_RATE).cuda()
    # _model = Delf_V1(ncls=n_classes,
    #                 arch='resnet34',
    #                 stage='finetune',
    #                 target_layer='layer3'
    #                 ).cuda()
    _model = DelfResNet().cuda()

    print('metric_fc')
    # Last Layer
    # _metric_fc = ArcMarginProduct(
    #    config.latent_dim, n_classes, s=config.S_TEMPERATURE, m=0.5, easy_margin=False).cuda()
    _metric_fc = nn.Linear(config.latent_dim, n_classes).cuda()
    # _metric_fc = DummyLayer().cuda()

    print('criterion')
    # Loss function
    # _criterion = nn.CrossEntropyLoss()
    _criterion = FocalLoss(gamma=2)

    print('optimizer')
    # Optimizer
    _optimizer = optim.Adam([{'params': _model.parameters()}, {
        'params': _metric_fc.parameters()}], lr=config.LEARNING_RATE)
    # _optimizer = optim.SGD([{'params': _model.parameters()}, {'params': _metric_fc.parameters()}],
    #                       lr=1e-3, momentum=0.9, weight_decay=1e-4)

    print('scheduler')
    # Scheduler
    mile_stones = [5, 7, 9, 10, 11, 12]
    _scheduler = optim.lr_scheduler.MultiStepLR(
        _optimizer, mile_stones, gamma=0.5, last_epoch=-1)
    # scheduler_step = EPOCHS
    # _scheduler = optim.lr_scheduler.CosineAnnealingLR(_optimizer, scheduler_step, eta_min=1e-4)

    return _model, _metric_fc, _criterion, _optimizer, _scheduler


def change_delf(_model, n_classes):
    """
    change Delf_V1(finetune) to keypoint

    Parameters
    ----------
    _model: Delf_V1
    """
    delf_state = {}
    _model.write_to(delf_state)
    model_new = Delf_V1(ncls=n_classes,
                        load_from=delf_state,
                        arch=_model.arch,
                        stage='keypoint',
                        target_layer='layer3',
                        use_random_gamma_rescale=True).cuda()
    return model_new


def train_main():
    get_logger().info('batch size: %d' % config.BATCH_SIZE_TRAIN)
    get_logger().info('epochs: %d' % config.EPOCHS)
    get_logger().info('latent_dim: %d' % config.latent_dim)
    get_logger().info('dropout_rate: %f' % config.DROPOUT_RATE)
    get_logger().info('scale temperature: %d' % config.S_TEMPERATURE)
    get_logger().info('the number of samples per class: %d' % config.N_SELECT)
    get_logger().info('the number of uniques for training: %d' % config.N_UNIQUES)
    get_logger().info('delf_resnet50')
    if config.USE_PRETRAINED:
        get_logger().info('pre-trained: %s' % config.PRETRAIN_PATH)

    # Train
    # load train data
    get_logger().info('loading df_train.')
    df_train = pd.read_csv(config.TRAIN_PATH, dtype={'id': 'object'})
    print(df_train.head())

    # create label encoder
    # must be init_le() before get_exist_image()
    if config.USE_PRETRAINED:
        le_path = os.path.join(config.PRETRAIN_PATH, 'le.pkl')
        get_logger().info('loading %s' % le_path)
        label_encoder = joblib.load(le_path)
    else:
        label_encoder = init_le(df_train)
        joblib.dump(label_encoder, 'le.pkl')

    # use landmark_id,  which is more than N_SELECT+1 images
    df_train = get_exist_image(df_train, config.TRAIN_IMG_PATH)
    df_train = select_train_data(df_train, config.N_UNIQUES)

    # train validate split
    df_trn, df_val = split_train_valid_v2(df_train)
    get_logger().info('train num: %d, valid num: %d' % (len(df_trn), len(df_val)))

    # initialize Dataset and DataLoader
    train_dataset = LandmarkDataset(config.TRAIN_IMG_PATH, df_trn,
                                    trn_trnsfms, mode='train', le=label_encoder)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True
                              )
    valid_dataset = LandmarkDataset(config.TRAIN_IMG_PATH, df_val,
                                    tst_trnsfms, mode='valid', le=label_encoder)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=False,
                              shuffle=False
                              )
    # Initialize model
    model, metric_fc, criterion, optimizer, scheduler \
        = init_model(label_encoder)

    # Start Training
    get_logger().info('start Training')
    start_epoch = 0
    best_score = 0
    # Load model
    if config.USE_PRETRAINED:
        start_epoch, model, metric_fc, optimizer, scheduler = \
            load_model(model, metric_fc, optimizer,
                       scheduler, reset_epoch=True)
        # change delf mode: finetune->keypoint
        n_classes = len(label_encoder.classes_)
        model = change_delf(model, n_classes)
        optimizer = optim.Adam([{'params': model.parameters()}, {
            'params': metric_fc.parameters()}], lr=config.LEARNING_RATE)
        mile_stones = [5, 7, 9, 10, 11, 12]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, mile_stones, gamma=0.5, last_epoch=-1)

    for epoch in range(start_epoch + 1, config.EPOCHS + 1):
        scheduler.step()

        tvp.train(epoch, model, train_loader,
                  metric_fc, criterion, optimizer)

        valid_score = tvp.validate_arcface(model, metric_fc, valid_loader)

        is_best = valid_score > best_score
        get_logger().info('best score (%f) at epoch (%d)' % (valid_score, epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'metric_fc': metric_fc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best)

        # update data
        train_loader.dataset.update()

    get_logger().info('end Training')


def predict_main():
    get_logger().info('batch size: %d' % config.BATCH_SIZE_TRAIN)
    get_logger().info('resnet34')
    get_logger().info('pre-trained: %s' % config.PRETRAIN_PATH)

    # load train data
    get_logger().info('loading df_test.')
    df_test_all = pd.read_csv(config.TEST_PATH, dtype={'id': 'object'})
    print(df_test_all.head())

    df_test = get_exist_image(df_test_all, config.TEST_IMG_PATH)

    test_dataset = LandmarkDataset(
        config.TEST_IMG_PATH, df_test, tst_trnsfms, mode='predict')
    label_encoder = joblib.load(os.path.join(config.PRETRAIN_PATH, 'le.pkl'))

    # Initialize model
    model, metric_fc, criterion, optimizer, scheduler \
        = init_model(label_encoder)

    # Load model
    start_epoch, model, metric_fc, optimizer, scheduler = \
        load_model(model, metric_fc, optimizer, scheduler)
    get_logger().info('prediction with a model of epoch: %d' % start_epoch)

    # Predict
    tvp.predict_label2(model, metric_fc, test_dataset, label_encoder)

    # load outputed file(submit_landmark.csv)
    get_logger().info('load outputed csv file.')
    df_sub = pd.read_csv('submit_landmark.csv', dtype={'id': 'object'})
    print(df_sub.head())
    get_logger().info('Shape of predicted csv: %s' % str(df_sub.shape))

    get_logger().info('load sample submit file.')
    df_sub_sample = pd.read_csv(config.SUBMIT_PATH, dtype={'id': 'object'})
    del df_sub_sample['landmarks']
    print(df_sub_sample.head())

    get_logger().info('merge df_sub_sample and df_sub')
    df_sub2 = df_sub_sample.merge(df_sub, how='left', on='id')[
        ['id', 'landmarks']]
    print(df_sub2.head())
    get_logger().info('Shape of submit csv file: %s' % str(df_sub2.shape))

    df_sub2['landmarks'].fillna('', inplace=True)
    get_logger().info('Sum of null value: %d' %
                      df_sub2['landmarks'].isnull().sum())

    get_logger().info('start writing submit_landmark2.csv')
    df_sub2.to_csv('submit_landmark2.csv', index=False)
    get_logger().info('end writing submit_landmark2.csv')


if __name__ == '__main__':
    create_logger('landmark.log')

    train_main()
    # predict_main()
