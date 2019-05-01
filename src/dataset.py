import os
import torch
from PIL import Image
from torch.utils.data import Dataset


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
        sub_folder0 = img_name[0]
        sub_folder1 = img_name[1]
        sub_folder2 = img_name[2]
        path = os.path.join(self.image_folder, sub_folder0,
                            sub_folder1, sub_folder2, img_name)
        # load images
        img = Image.open(path).convert('RGB')
        return img
