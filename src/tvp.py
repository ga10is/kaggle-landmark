from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config
from .common.util import debug_trace
from .model.metrics import AverageMeter, accuracy
from .model.loss import ArcMarginProduct
from .common.logger import get_logger
from .common.util import str_stats


@debug_trace
def train(epoch,
          model,
          loader,
          metric_fc,
          criterion,
          optimizer):
    """
    1 epoch training

    Parameters
    ----------
    epoch: int
        epoch of the train
    model: nn.Module
    loader: DataLoader
    metric_fc: nn.Module
        layer before loss function
    criterion: nn.Module
        loss function
    optimizer: Optimizer
    """
    loss_meter = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    get_logger().info('[Start] epoch: %d' % epoch)
    get_logger().info('lr: %f' %
                      optimizer.state_dict()['param_groups'][0]['lr'])

    softmax = torch.nn.Softmax(dim=1).cuda()
    # train phase
    model.train()
    for i, data in enumerate(tqdm(loader)):
        img, label = data
        img, label = img.cuda(), label.cuda().long()
        with torch.set_grad_enabled(True):
            # forward
            emb_vec = model(img)
            if isinstance(metric_fc, ArcMarginProduct):
                # ArcMarginProduct needs label when training
                logit = metric_fc(emb_vec, label)
            else:
                logit = metric_fc(emb_vec)
            loss = criterion(logit, label.squeeze())

            # measure accuracy
            prec1, prec5 = accuracy(logit.detach(), label, topk=(1, 5))
            loss_meter.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print
        if i % config.PRINT_FREQ == 0:
            logit_cpu = logit.detach().cpu()
            get_logger().info('\n' + str_stats(logit_cpu[0].numpy()))
            softmaxed = softmax(logit_cpu)
            get_logger().info('\n' + str_stats(softmaxed[0].numpy()))
            get_logger().info('train: %d loss: %f top1: %f top5: %f (just now)' %
                              (i, loss_meter.val, top1.val, top5.val))
            get_logger().info('train: %d loss: %f top1: %f top5: %f' %
                              (i, loss_meter.avg, top1.avg, top5.avg))
    get_logger().info(
        "Epoch %d/%d train loss %f top1 %f top5 %f" % (epoch, config.EPOCHS, loss_meter.avg, top1.avg, top5.avg))

    return loss_meter.avg


def validate_arcface(model,
                     metric_fc,
                     loader):
    """
    validation

    Parameters
    ----------
    model: nn.Module
    metric_fc: nn.Module
    loader: DataLoader
    """
    top1 = AverageMeter()
    top5 = AverageMeter()

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        img, label = data
        img, label = img.cuda(), label.cuda().long()
        with torch.no_grad():
            # forward
            emb_vec = model(img)
            logit = metric_fc(emb_vec)

        # measure accuracy
        prec1, prec5 = accuracy(logit.detach(), label, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        # print
        if i % config.PRINT_FREQ == 0:
            get_logger().info('valid: %d top1: %f top5: %f (just now)' %
                              (i, top1.val, top5.val))
            get_logger().info('valid: %d top1: %f top5: %f' %
                              (i, top1.avg, top5.avg))

    get_logger().info("valid top1 %f top5 %f" % (top1.avg, top5.avg))

    return top1.avg


def make_df(df_org, labels, confidences, le):
    """
    make dataframe for submission

    Parameters
    ----------
    df_org: pd.DataFrame of shape = [n_samples, more than 1]
        dataframe which have id column
    labels: ndarray of shape = [n_samples, n_predicts]
        array of indices(number)
    confidences: ndarray of shape = [n_samples, n_predicts]
        array of confidence(float)
    le: LabelEncoder
        label encoder

    Returns
    ------
    pd.DataFrame of shape = [n_samples, 2]
        the dataframe has 'id', 'landmarks' columns.
    """
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    new_df = pd.DataFrame()
    new_df['id'] = df_org['id']
    new_df['landmarks'] = ''
    for i in range(labels.shape[1]):
        if i != 0:
            new_df['landmarks'] += ' '
        new_df['landmarks'] += le.inverse_transform(labels[:, i]).astype(str)
        new_df['landmarks'] += ' '
        new_df['landmarks'] += confidences[:, i].astype(str)

    return new_df


def predict_label(model, metric_fc, test_dataset, label_encoder):
    """
    prediction

    Paramters
    ---------
    model: nn.Module
    metric_fc: nn.Module
    test_dataset: Dataset
    label_encoder: LabelEncoder
    """
    pred_indices = []
    pred_confs = []

    loader = DataLoader(test_dataset,
                        batch_size=config.BATCH_SIZE_TRAIN,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=True,
                        drop_last=False,
                        shuffle=False
                        )

    softmax = torch.nn.Softmax(dim=1).cuda()
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        img, _ = data

        with torch.no_grad():
            if not config.RUN_TTA:
                img = img.cuda()
                # forward
                emb_vec = model(img)
                confs = metric_fc(emb_vec)
                if not isinstance(metric_fc, ArcMarginProduct):
                    # Calculate confidences by softmax if loss is not ArcFace.
                    # Use cosine similarity as confidences if loss is ArcFace.
                    # The similarity is computed by ArcMarginProduct.
                    confs = softmax(confs)
            else:
                # When TTA, DataLoader returns list of torch.Tensor
                imgs = img
                img = imgs[0].cuda()
                confs = metric_fc(model(img))
                if not isinstance(metric_fc, ArcMarginProduct):
                    confs = softmax(confs)
                sum_confs = confs

                for i in range(1, len(imgs)):
                    img = imgs[i].cuda()
                    confs = metric_fc(model(img))
                    if not isinstance(metric_fc, ArcMarginProduct):
                        confs = softmax(confs)
                    sum_confs += confs
                confs = sum_confs / len(imgs)

            top_confs, top_indices = torch.topk(confs, k=20)
            top_indices = top_indices.detach().cpu().numpy()
            top_confs = top_confs.detach().cpu().numpy()

            pred_indices.append(top_indices)
            pred_confs.append(top_confs)

    pred_indices = np.concatenate(pred_indices)
    pred_confs = np.concatenate(pred_confs)

    # make df
    df_submit = make_df(loader.dataset._df, pred_indices,
                        pred_confs, label_encoder)

    # write result
    submit_file = 'submit_landmark.csv'
    df_submit.to_csv(submit_file, index=False)
