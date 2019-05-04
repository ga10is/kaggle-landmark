import sys
import os
import multiprocessing
import csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import tqdm

from .. import config


def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def download_image(key_url):
    # out_dir = sys.argv[2]
    out_dir = config.TEST_IMG_PATH
    (key, url) = key_url
    # filename = os.path.join(out_dir, '{}.jpg'.format(key))
    sub_dir0 = key[0]
    sub_dir1 = key[1]
    sub_dir2 = key[2]
    sub_dir_path = os.path.join(out_dir, sub_dir0, sub_dir1, sub_dir2)
    filename = os.path.join(sub_dir_path, '{}.jpg'.format(key))

    os.makedirs(sub_dir_path, exist_ok=True)

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        # print('.', end='')
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data)).resize(config.IMG_SIZE)
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1

    return 0


def loader():
    '''
    if len(sys.argv) != 3:
        print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    '''
    # (data_file, out_dir) = sys.argv[1:]
    # (data_file, out_dir) = TRAIN_PATH, TRAIN_IMG_PATH
    (data_file, out_dir) = config.TEST_PATH, config.TEST_IMG_PATH

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=8)  # Num of CPUs
    failures = sum(tqdm.tqdm(pool.imap_unordered(
        download_image, key_url_list), total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


if __name__ == '__main__':
    loader()
    # arg1 : data_file.csv
    # arg2 : output_dir
    # if __name__ == '__main__':
    #     loader()
