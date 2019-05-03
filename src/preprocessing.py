import shutil
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import LabelEncoder

from .dataset import first_selection


def get_exist_image(_df, _image_folder):
    """
    create dataframe of exist images in folder
    """
    exist_images = set(get_image_ids(_image_folder))
    df_exist = _df[_df['id'].isin(exist_images)]
    print(len(exist_images))
    print('exist images: %d' % len(df_exist))
    return df_exist


def assert_exist_image(df, image_folder):
    exist_images = set(get_image_ids(image_folder))
    df_image = set(df['id'].values)
    print(len(exist_images))
    print(len(df_image))
    assert (exist_images == df_image), \
        'There are not all images in the "image_folder"'


def get_image_ids_from_subdir(_dir_path, _sub_dir):
    result = []

    sub_dir0 = _sub_dir
    sub_dir_path0 = os.path.join(_dir_path, sub_dir0)
    for sub_dir1 in os.listdir(sub_dir_path0):
        sub_dir_path1 = os.path.join(sub_dir_path0, sub_dir1)
        for sub_dir2 in os.listdir(sub_dir_path1):
            sub_dir_path2 = os.path.join(sub_dir_path1, sub_dir2)
            image_ids = [image_file.split('.')[0]
                         for image_file in os.listdir(sub_dir_path2)]
            result.extend(image_ids)
    return result


def get_image_ids(dir_path):
    result = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for sub_dir in os.listdir(dir_path):
            sub_dir_path = os.path.join(dir_path, sub_dir)
            if os.path.isdir(sub_dir_path):
                print('dir: %s' % sub_dir_path)
                futures.append(executor.submit(
                    get_image_ids_from_subdir, dir_path, sub_dir))

        for future in tqdm(futures):
            result.extend(future.result())
    return result


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


def select_train_data(df_train, n_duplicated):
    """
    Select train data.
    Select landmark_id, which is more than 'n' images.
    """
    id_count = df_train['landmark_id'].value_counts()
    s_count = df_train['landmark_id'].map(id_count)
    df_train = df_train[s_count >= n_duplicated]
    print('landmark_id >= %d: %d, images: %d' %
          (n_duplicated, (id_count >= n_duplicated).sum(), len(df_train)))
    return df_train


def split_train_valid_v2(df):
    print('start select_train_valid_v2()')
    # df_valid = random_selection(df, 1)
    df_valid = first_selection(df)
    valid_ids = df_valid['id'].values
    df_train = df[~df['id'].isin(valid_ids)]
    print('end select_train_valid_v2()')

    return df_train, df_valid
