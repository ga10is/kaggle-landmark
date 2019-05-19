import os
import pandas as pd
from tqdm import tqdm


def move_image():
    csv_path = '/home/galois/landmark/data/categories_places365_extended.csv'
    dir_path = '/mnt/disks/extend/places365_standard/train'

    df = pd.read_csv(csv_path)
    df_non = df.loc[df['io'] == 1]
    print(df_non.head())
    print(df_non.shape)

    def trim(s):
        return s[3:]

    df_non['dir'] = df_non['label'].trainsform(trim)
    print(df_non.head())


def make_id_label(image_name, label):
    id_name = 'non-%s@%s' % (label, image_name.split('.')[0])
    return [id_name, label]


def make_non_landmark_df():
    """
    Returns
    -------
    non_landmark: pd.DataFrame
        dataframe which has 'id' and 'landmark_id' columns
    """
    path = './data/places365/train/places365_indoor'

    all_records = []
    for label in tqdm(os.listdir(path)):
        dir_path = os.path.join(path, label)
        records = [make_id_label(image_name, label)
                   for image_name in os.listdir(dir_path)]
        all_records.extend(records)

    non_landmark = pd.DataFrame(
        data=all_records, columns=['id', 'landmark_id'])
    print(non_landmark.head())

    non_landmark.to_csv('non_landmark.csv', index=False)
    return non_landmark


if __name__ == '__main__':
    make_non_landmark_df()
