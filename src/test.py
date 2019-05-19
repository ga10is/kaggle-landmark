import pandas as pd
import random
import numpy as np


def func_landmark_to_id(group, n_samples):
    def func(landmark):
        return random.sample(group.get_group(landmark).tolist(), n_samples)
    return func


def func2_landmark_to_id(landmark, group, n_sample):
    sampled = random.sample(group.get_group(landmark).tolist(), n_sample)
    return np.array(sampled)


np_func2_landmark_to_id = np.vectorize(
    func2_landmark_to_id, excluded=['group', 'n_sample'], signature='()->(n)')


def make_id_label(image_name, label):
    id_name = 'non-%s@%s' % (label, image_name.split('.')[0])
    return [id_name, label]


def test_randmselect():
    df = pd.DataFrame({
        'id': np.array(range(10)).astype(str),
        'landmark_id': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
    })
    g = df.groupby('landmark_id')['id']
    df_new = g.first().resent_index()
    # selected_images = g.first().values
    # df_new = pd.DataFrame({'id': selected_images})
    # df_new = df_new.merge(df[['id', 'landmark_id']], on='id', how='left')
    print(df)

    landmark2id = func_landmark_to_id(g, 3)

    """
    landmark_list = df['landmark_id'].unique().tolist()
    selected_images = [landmark2id(landmark_id)
                       for landmark_id in landmark_list]
    selected_images = np.array(selected_images).flatten()
    """
    landmark_array = df['landmark_id'].unique()
    selected_images = np_func2_landmark_to_id(
        landmark_array, group=g, n_sample=2)
    selected_images = selected_images.reshape(-1)

    df_new = pd.DataFrame({'id': selected_images})
    df_new = df_new.merge(df[['id', 'landmark_id']], on='id', how='left')
    print(df)


def make_non_landmark_df():
    images = ['001.jpg', '002.jpg', '003.jpg']
    labels = ['airport', 'airplane-indoor']

    all_records = []
    for label in labels:
        records = [make_id_label(image_name, label) for image_name in images]
        all_records.extend(records)

    df_non_land = pd.DataFrame(data=all_records, columns=['id', 'landmark_id'])
    print(df_non_land)

    name_list = df_non_land['id'].tolist()
    for image_name in name_list:
        image_name = '%s.jpg' % image_name
        if image_name.startswith('non-'):
            image_name = image_name[4:]
            dir_name, file_name = image_name.split('@')
            print('%s/%s' % (dir_name, file_name))


def trans_label(s):
    s_new = s[3:]
    s_new = s_new.replace('/', '-')
    return s_new


PLACES365_PATH = './data/preprocess/categories_places365_extended.csv'


def remove_non_landmark(submit):
    """
    Replace 'landmarks' column in submit dataframe to '' if the column is are predicted as non landmark.

    Parameters
    ----------
    submit: pd.DataFrame
        submit dataframe whichi has 'landmarks' column

    Returns
    -------
    submit: pd.DataFrame
        replaced dataframe
    """
    df = pd.read_csv(PLACES365_PATH)
    non_landmark = df[df['io'] == 1]
    non_landmark_list = non_landmark['label'].transform(trans_label)

    s_label = submit['landmarks'].str.split(' ', expand=True)[0]
    is_non_landmark = s_label.isin(non_landmark_list)
    submit.loc[is_non_landmark, 'landmarks'] = ''

    return submit


def call_remove_non_landmark():
    submit = pd.DataFrame({
        'id': np.array(range(5)).astype(str),
        'landmarks': [
            '1234 0.1', 'auto_showroom 0.2', '135 0.5', 'bazaar-indoor 0.5', '345 0.5'
        ]
    })
    print(submit)
    remove_non_landmark(submit)
    print(submit)


if __name__ == '__main__':
    call_remove_non_landmark()
