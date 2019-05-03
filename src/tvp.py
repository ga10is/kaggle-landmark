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


@debug_trace
def train(epoch,
          model,
          loader,
          metric_fc,
          criterion,
          optimizer):
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
        # label = label.squeeze() # (batch_size, 1) -> (batch_size,)
        # optimizer.zero_grad()
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
            softmaxed = softmax(logit.detach().cpu())
            get_logger().info(torch.max(softmaxed[0]).item())
            get_logger().info(softmaxed[0, 0:10])
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

        '''
        # calculate accuracy
        max_proba = np.max(proba, axis=1)
        max_proba_idx = np.argmax(proba, axis=1)
        acc = accuracy_score(label, max_proba_idx)
        # calculate GAP
        gap = GAP_vector(max_proba_idx, max_proba, label.squeeze())
        get_logger().info("validate score: acc %f gap %f" % (acc, gap))
        '''

    get_logger().info("valid top1 %f top5 %f" % (top1.avg, top5.avg))

    return top1.avg


def make_df(df_org, labels, confidences):
    """
    make dataframe for submission
    df_org: pd.DataFrame of shape = [n_samples, more than 1]
        dataframe which have id column
    labels: ndarray of shape = [n_samples]
        array of label(number)
    confidences: ndarray of shape = [n_samples]
        array of confidence(float)
    Returns
    pd.DataFrame of shape = [n_samples, 2]
        the dataframe has 'id', 'landmarks' columns.
    """
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    new_df = pd.DataFrame()
    new_df['id'] = df_org['id']
    new_df['label'] = labels.astype(str)
    new_df['confidence'] = confidences.astype(str)
    new_df['landmarks'] = new_df['label'] + ' ' + new_df['confidence']
    del new_df['label'], new_df['confidence']
    return new_df


def predict_label2(model, metric_fc, test_dataset, label_encoder):

    pred_indices = []
    pred_scores = []
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
        img = img.cuda()
        with torch.no_grad():
            # forward
            emb_vec = model(img)
            logit = metric_fc(emb_vec)

            # from IPython.core.debugger import Pdb; Pdb().set_trace()
            top_scores, top_indices = torch.topk(logit, k=20)
            top_indices = top_indices.detach().cpu().numpy()
            top_scores = top_scores.detach().cpu().numpy()

            confs = softmax(logit)
            top_confs, _ = torch.topk(confs, k=20)
            top_confs = top_confs.detach().cpu().numpy()

            pred_indices.append(top_indices)
            pred_scores.append(top_scores)
            pred_confs.append(top_confs)

    pred_indices = np.concatenate(pred_indices)
    pred_scores = np.concatenate(pred_scores)
    pred_confs = np.concatenate(pred_confs)

    # make df
    labels = label_encoder.inverse_transform(pred_indices[:, 0])
    df_submit = make_df(loader.dataset.df, labels, pred_confs[:, 0])

    # write result
    submit_file = 'submit_landmark.csv'
    df_submit.to_csv(submit_file, index=False)
