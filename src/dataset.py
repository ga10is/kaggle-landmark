import os
import numpy as np
import pandas as pd
import random
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config
from .common.logger import get_logger


class LandmarkDataset(Dataset):
    def __init__(self, image_folder, df, transform, mode, le=None):
        self.image_folder = image_folder
        self.transform = transform
        self._df = df
        self.mode = mode
        if self.mode == 'train':
            # selected images
            self.selected = random_selection(df, config.N_SELECT)
        else:
            self.selected = self._df

        if self.mode in ['train', 'valid']:
            # Label Encoder
            if le is None:
                raise ValueError(
                    'Argument "le" must not be None when "mode" is train or valid.')
            self.le = le

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, idx):
        img_name = '%s.jpg' % self.selected.iloc[idx]['id']
        if img_name.startswith('non-'):
            # if image is not landmark
            img = self.__get_image_non_landmark(img_name)
        else:
            img = self.__get_image(img_name)

        label = None
        if self.mode in ['train', 'valid']:
            id = self.selected.iloc[idx]['landmark_id']
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
        load images
        """
        sub_folder0 = img_name[0]
        sub_folder1 = img_name[1]
        sub_folder2 = img_name[2]
        path = os.path.join(self.image_folder, sub_folder0,
                            sub_folder1, sub_folder2, img_name)
        # load images
        img = Image.open(path).convert('RGB')
        return img

    def __get_image_non_landmark(self, img_name):
        """
        load no landmark image
        """
        img_name = img_name[4:]
        dir_name, file_name = img_name.split('@')
        path = os.path.join(config.NON_LANDMARK_IMG_PATH, dir_name, file_name)

        # load image
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

    def update(self):
        """
        Update self.selected.
        """
        if self.mode != 'train':
            raise ValueError('LandmarkDataset is not train mode.')
        self.selected = random_selection(self._df, config.N_SELECT)
        print('updated selected images.')


def func_landmark_to_id(group, n_samples):
    def func(landmark):
        return random.sample(group.get_group(landmark).tolist(), n_samples)
    return func


def func2_landmark_to_id(landmark, group, n_sample):
    """
    Select id of sample images according to landmark_id.

    Parameters
    ----------
    landmark: str
        landmark_id
    group: pandas.GroupBy
    n_sample: int
        the number of samples
    """
    sampled = random.sample(group.get_group(landmark).tolist(), n_sample)
    # specify the length of string because numpy cut string for aligning the length of the string
    return np.array(sampled, dtype='U50')


# vectorized function
np_func2_landmark_to_id = np.vectorize(
    func2_landmark_to_id, excluded=['group', 'n_sample'], signature='()->(n)')


def random_selection(_df, n_samples):
    """
    Select images randomly from _df.
    """
    print('selecting (%d) samples from (%d) iamges.' % (n_samples, len(_df)))
    g = _df.groupby('landmark_id')['id']

    """
    landmark2id = func_landmark_to_id(g, n_samples)
    landmark_list = _df['landmark_id'].unique().tolist()
    selected_images = [landmark2id(landmark_id)
                       for landmark_id in landmark_list]
    selected_images = np.array(selected_images).flatten()
    """
    landmark_array = _df['landmark_id'].unique()
    selected_images = np_func2_landmark_to_id(
        landmark_array, group=g, n_sample=n_samples)
    selected_images = selected_images.reshape(-1)
    print('num of selected_images: %d' % len(selected_images))

    df_new = pd.DataFrame({'id': selected_images})
    df_new = df_new.merge(
        _df[['id', 'landmark_id']], on='id', how='left')

    print('num df_new: %d' % len(df_new))
    print(df_new.head())

    # assertion value_counts()

    return df_new


def first_selection(_df):
    """
    Select first images for each group from _df.
    """
    get_logger().info('selecting first sample from (%d) iamges.' % (len(_df)))
    g = _df.groupby('landmark_id')['id']
    # selected_images = g.first().values

    # print('num of selected_images: %d' % len(selected_images))

    """
    df_new = pd.DataFrame({'id': selected_images})
    df_new = df_new.merge(
        _df[['id', 'landmark_id']], on='id', how='left')
    """
    df_new = g.first().reset_index()

    print('num df_new: %d' % len(df_new))
    print(df_new.head())

    return df_new
