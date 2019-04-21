# -*- coding: utf-8 -*-
"""landmark_arcface.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RBtQv0ilYMer39d4NOEmKp4_i0JdFz9h
"""

from google.colab import drive
drive.mount('/gdrive')

# %cd "/gdrive/My Drive"

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
import joblib

IMG_SIZE = (256, 256)
INPUT = 'analysis/landmark/data/raw/'
INDEX_PATH = INPUT + 'index.csv'
TRAIN_PATH = INPUT + 'train.csv'
TEST_PATH = INPUT + 'test.csv'
TRAIN_IMG_PATH = INPUT + 'train/'
TEST_IMG_PATH = INPUT + 'test/'
INDEX_IMG_PATH = INPUT + 'index/'

"""## utility"""

def debug_deco(func):
    def wrapper(*args, **kwargs):
        print('--start--')
        from IPython.core.debugger import Pdb; Pdb().set_trace()
        func(*args, **kwargs)
        print('--end--')
    return wrapper

"""## logging"""

import logging

def create_logger(log_file_name):
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    #fh = logging.FileHandler('whale.log')
    fh = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=100000, backupCount=8)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger():
    return logging.getLogger('main')

create_logger('landmark.log')

"""## preprocessing"""

import os 
from concurrent.futures import ProcessPoolExecutor

def get_exist_image(_df, _image_folder):
    """
    create dataframe of exist images in folder
    """
    exist_images = get_image_ids(_image_folder)
    df_exist = _df[_df['id'].isin(exist_images)]
    print('exist images: %d' % len(exist_images))
    return df_exist

def assert_exist_image(df, image_folder):
    exist_images = set(get_image_ids(image_folder))
    df_image = set(df['id'].values)
    print(len(exist_images))
    print(len(df_image))
    assert (exist_images == df_image), 'There are not all images in the "image_folder"'


def get_image_ids_from_subdir(_dir_path, _sub_dir):
    sub_dir_path = os.path.join(_dir_path, _sub_dir)
    image_ids = [image_file.split('.')[0] for image_file in os.listdir(sub_dir_path)]
    return image_ids


def get_image_ids(dir_path):
    result = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for sub_dir in os.listdir(dir_path):
            futures.append(
                executor.submit(get_image_ids_from_subdir, dir_path, sub_dir))

        for future in tqdm(futures):
            result.extend(future.result())
    return result


import shutil


def move_to_folder(dir_path):
    for file in tqdm(os.listdir(dir_path)):
        if(file[-4:] == '.jpg'):
            # move image
            sub_dir = file[0:2]
            sub_dir_path = os.path.join(dir_path, sub_dir)
            old_path = os.path.join(dir_path, file)
            new_path = os.path.join(dir_path, sub_dir, file)
            
            os.makedirs(sub_dir_path, exist_ok=True)
            
            shutil.move(old_path, new_path)
        else:
            print('There is a file which is not image: %s' % file)
            

def init_le(_df):
    ids = _df['landmark_id'].values.tolist()
    le = LabelEncoder()
    le.fit(ids)
    return le

# move_to_folder(TEST_IMG_PATH)

"""## nn"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms

def imshow(img):
    #print(type(img))
    img = img * 0.23 + 0.5     # unnormalize
    npimg = img.numpy()
    #print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def save_checkpoint(state, is_best, fpath='checkpoint.pth'):
    torch.save(state, fpath)
    if is_best:
        torch.save(state, 'best_model.pth')
        
def load_checkpoint(_model, 
                    _metric_fc,
                    _optimizer, 
                    _scheduler, 
                    fpath):
    checkpoint = torch.load(fpath)
    _epoch = checkpoint['epoch']
    _model.load_state_dict(checkpoint['state_dict'])
    _metric_fc.load_state_dict(checkpoint['metric_fc'])
    _optimizer.load_state_dict(checkpoint['optimizer'])
    _scheduler.load_state_dict(checkpoint['scheduler'])
    
    return _epoch, _model, _metric_fc, _optimizer, _scheduler

trn_trnsfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    #transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=(-30,30), shear=(-30,30)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

tst_trnsfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ResNet(nn.Module):
    def __init__(self, output_neurons, n_classes, dropout_rate):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)        
        #self.resnet = torchvision.models.resnet18(pretrained=True)
        self.norm1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout_rate)
        # FC
        self.fc = nn.Linear(512, output_neurons)
        self.norm2 = nn.BatchNorm1d(output_neurons)
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # GAP
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.norm1(x)
        x = self.drop1(x)
        # FC
        x = self.fc(x)
        x = self.norm2(x)
        #x = l2_norm(x)
        return x
    
class DenseNet(nn.Module):
    def __init__(self, output_neurons, n_classes, dropout_rate):
        super(DenseNet, self).__init__()
        self.densenet_features = torchvision.models.densenet121(pretrained=True).features
        self.norm1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1024, output_neurons)
        self.norm2 = nn.BatchNorm1d(output_neurons)
        
    def forward(self, x):
        features = self.densenet_features(x)
        x = F.relu(features, inplace=True)
        # GAP
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(features.size(0), -1)
        x = self.norm1(x)
        x = self.drop1(x)
        # FC
        x = self.fc(x)
        x = self.norm2(x)
        return x

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        #print(output[0])

        return output
    
class FocalBinaryLoss(nn.Module):
    def __init__(self, gamma=0):
        super(FocalBinaryLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        p = torch.sigmoid(input)        
        loss = torch.mean(-1 * target * torch.pow(1-p, self.gamma) * torch.log(p + 1e-10) +
                          -1 * (1-target) * torch.pow(p, self.gamma) * torch.log(1-p + 1e-10)) * 4
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

"""## dataset"""

from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader,Dataset

class LandmarkDataset(Dataset):
    def __init__(self, image_folder, df, transform, is_train, le=None):
        self.image_folder = image_folder  
        self.transform = transform      
        self.df = df
        self.is_train = is_train
        if is_train:
            if le is None:
                raise ValueError(
                    'Argument "le" must not be None when "is_train" is True.')
            self.le = le
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_name = '%s.jpg' % self.df.iloc[idx]['id']                   
        img = self.__get_image(img_name)
        label = None
        if self.is_train:
            id = self.df.iloc[idx]['landmark_id']
            label = torch.tensor(self.le.transform([id]))
        else:
            label = -1
        return img, label
    
    def __get_image(self, img_name):           
        img = self.__load_image(img_name)
        img = self.transform(img)
        return img

    def __load_image(self, img_name):
        """
        load images and bound boxing
        """
        sub_folder = img_name[0:2]
        path = os.path.join(self.image_folder, sub_folder, img_name)
        # load images
        img = Image.open(path).convert('RGB')               
        return img

"""## metrics"""

def GAP_vector(pred, conf, true, return_x=False):
    '''
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition.
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    '''
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

"""## train_valid"""

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
    get_logger().info('lr: %f' % optimizer.state_dict()['param_groups'][0]['lr'])
        
    # train phase
    model.train()
    for i, data in enumerate(tqdm(loader)):
        img, label = data                
        img, label = img.cuda(), label.cuda().long()        
        #label = label.squeeze() # (batch_size, 1) -> (batch_size,)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward                      
            emb_vec = model(img)
            logit = metric_fc(emb_vec, label)
            loss = criterion(logit, label.squeeze())

            # backward
            loss.backward()
            optimizer.step()            
            
        # measure accuracy
        prec1, prec5 = accuracy(logit.detach(), label, topk=(1, 5))
        loss_meter.update(loss.item(), img.size(0))
        top1.update(prec1[0], img.size(0))
        top5.update(prec5[0], img.size(0))
        
        # print
        if i % PRINT_FREQ == 0:
            print('loss: %f top1: %f top5: %f' % (loss_meter.avg, top1.avg, top5.avg))
    get_logger().info(
        "Epoch %d/%d train loss %f" % (epoch, EPOCHS, loss_meter.avg))

    # update pairs of image
    get_logger().info('Finished updating dataset')
    return loss_meter.avg


def validate_arcface(model,
                     metric_fc,
                     unknown_loader,
                     label_encoder
                    ):
    """

    """
    # validate phase
    model.eval()    
    with torch.no_grad():       
        # predict latent features of Unknown whales
        uk_output = predict_arcface(model, metric_fc, unknown_loader)        
        n_predict = 1
        for i in range(n_predict-1):
            uk_output2 = predict_arcface(model, metric_fc, unknown_loader)
            print(uk_output2[:5, :5])
            uk_output += uk_output2
        uk_output /= n_predict
        
        df_known = make_df(label_encoder)
        mat_distance = pd.DataFrame(data=uk_output.T, 
                                    columns=unknown_loader.dataset.df['Image'].values, 
                                    index=df_known['Image'].values)

        # compute map@5
        score = compute_map5(mat_distance, df_known, unknown_loader.dataset.df)        
        #acc = accuracy(uclasses, unknown_loader.dataset.df['Id'].values)
        get_logger().info("validate score %f" % (score))

    return score, mat_distance

def predict_proba(model, metric_fc, loader):
    """
    return Tensor of probability for each class
    """
    outputs = []
    for data in tqdm(loader):
        model.eval()
        with torch.no_grad():
            img, _ = data
            img = img.cuda()        
            output = metric_fc(model(img))
            outputs.append(output.detach().cpu().numpy())    
    outputs = np.concatenate(outputs)
    return outputs



def split_dataset(dataset, steps):
    """
    split Dataset by steps and create DataLoader.
    Parameters
    dataset: torch.utils.data.Dataset
    steps: int
        the number of each dataset    
    Returns
    list of torch.utils.data.DataLoader
    """
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    _df = dataset.df
    n = _df.shape[0]
    loader_list = []
    
    split_indexes= np.array_split(np.arange(n), steps)
    for split_index in split_indexes:
        split_df = _df.iloc[split_index]
        split_dataset = LandmarkDataset(dataset.image_folder, 
                                        split_df, 
                                        dataset.transform, 
                                        is_train=False)
        split_loader = DataLoader(split_dataset,
                          batch_size=BATCH_SIZE_TRAIN,
                          num_workers=NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False,
                          shuffle=False
                         )
        loader_list.append(split_loader)
    return loader_list

    

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


def predict_label(model, metric_fc, test_dataset, label_encoder):
    submit_file = 'submit_landmark.csv'
    
    # split df in test_dataset and make loader
    loaders = split_dataset(test_dataset, 10)
    
    # write the header of a submission table
    df_header = pd.DataFrame(columns=['id', 'landmarks'])
    df_header.to_csv(submit_file, index=False)
    
    # prediction phase
    for i, loader in enumerate(loaders):
        get_logger().info('prediction %d / %d' % (i+1, len(loaders)))
        model.eval()
        with torch.no_grad():    
            proba = predict_proba(model, metric_fc, loaders[i])
            max_proba = np.max(proba, axis=1)
            max_proba_idx = np.argmax(proba, axis=1)
            labels = label_encoder.inverse_transform(max_proba_idx)

            df_submit = make_df(loader.dataset.df, labels, max_proba)

        # write result in appending mode
        df_submit.to_csv(submit_file, index=False, header=False, mode='a')
        
    get_logger().info("created submission file")

"""## main"""

BATCH_SIZE_TRAIN = 100
NUM_WORKERS = 8
EPOCHS = 10
PRINT_FREQ = 10

"""### train"""

df_train = pd.read_csv(TRAIN_PATH)
df_train.head()

# must be init_le() before get_exist_image()
label_encoder = init_le(df_train)
joblib.dump(label_encoder, 'le.pkl')

df_train = get_exist_image(df_train, TRAIN_IMG_PATH)

train_dataset = LandmarkDataset(TRAIN_IMG_PATH, df_train, 
                                trn_trnsfms, is_train=True, le=label_encoder)
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE_TRAIN,
                          num_workers=NUM_WORKERS,
                          pin_memory=True,
                          drop_last=True,
                          shuffle=True
                         )

counting = train_dataset.df['landmark_id'].value_counts()
(counting > 1).sum()

train_dataset.df.shape

latent_dim = 512

n_classes = len(label_encoder.classes_)
#model = DenseNet(output_neurons=latent_dim, n_classes=len(le.classes_),  dropout_rate=0.5).cuda()    
model = ResNet(output_neurons=latent_dim, n_classes=n_classes, dropout_rate=0.5).cuda()    
metric_fc = ArcMarginProduct(latent_dim, n_classes, s=60, m=0.5, easy_margin=False).cuda()
criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(gamma=2)

#optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.SGD([{'params':model.parameters()}, {'params': metric_fc.parameters()}], 
                      lr=5e-3, momentum=0.9, weight_decay=1e-4)
scheduler_step = EPOCHS
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, eta_min=1e-4)

start_epoch = 0

for epoch in range(start_epoch+1, EPOCHS+1):
    scheduler.step()
    
    epoch_loss = train(epoch, model, train_loader, metric_fc, criterion, optimizer)

save_checkpoint({
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'metric_fc': metric_fc.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
}, True)

"""### prediction"""

df_test_all = pd.read_csv(TEST_PATH)
df_test_all.head()

df_test = get_exist_image(df_test_all, TEST_IMG_PATH)

test_dataset = LandmarkDataset(TEST_IMG_PATH, df_test, tst_trnsfms, is_train=False)
label_encoder = joblib.load('analysis/landmark/models/le.pkl')

len(label_encoder.classes_)

latent_dim = 512

n_classes = len(label_encoder.classes_)
#model = DenseNet(output_neurons=latent_dim, n_classes=len(le.classes_),  dropout_rate=0.5).cuda()    
model = ResNet(output_neurons=latent_dim, n_classes=n_classes, dropout_rate=0.5).cuda()    
metric_fc = ArcMarginProduct(latent_dim, n_classes, s=60, m=0.5, easy_margin=False).cuda()
criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(gamma=2)

#optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.SGD([{'params':model.parameters()}, {'params': metric_fc.parameters()}], 
                      lr=1e-3, momentum=0.9, weight_decay=1e-4)
scheduler_step = 200
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, eta_min=1e-4)

# create model object before following statement
start_epoch, model, metric_fc, optimizer, scheduler = load_checkpoint(model, 
                                                                      metric_fc,
                                                                      optimizer,
                                                                      scheduler, 
                                                                      'analysis/landmark/models/best_model.pth')

predict_label(model, metric_fc, test_dataset, label_encoder)

df_sub = pd.read_csv('submit_landmark.csv')
df_sub.shape

df_test_all.shape

df_sub.head()

df_test_all.head()

df_sub2 = df_test_all.merge(df_sub, how='left', on='id')[['id', 'landmarks']]
df_sub2.shape

df_sub2.head()

df_sub2['landmarks'].fillna('', inplace=True)
df_sub2['landmarks'].isnull().sum()

df_sub2.to_csv('submit_landmark2.csv', index=False)
