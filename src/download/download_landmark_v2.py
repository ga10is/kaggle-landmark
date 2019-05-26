import os
import time
import hashlib
import tarfile
import urllib.request
from functools import partial
from multiprocessing import Pool

import cv2
from tqdm import tqdm

# images_base_url = 'https://s3.amazonaws.com/google-landmark/train/images_{:03d}.tar'
# md5_base_url = 'https://s3.amazonaws.com/google-landmark/md5sum/train/md5.images_{:03d}.txt'
images_base_url = 'https://s3.amazonaws.com/google-landmark/test/images_{:03d}.tar'
md5_base_url = 'https://s3.amazonaws.com/google-landmark/md5sum/test/md5.images_{:03d}.txt'


def md5(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def process_image(file, source_dir, target_dir):
    source_path = os.path.join(source_dir, file)
    image = cv2.imread(source_path)
    # image = cv2.resize(image, (448, 448))
    image = cv2.resize(image, (256, 256))
    target_path = source_path.replace(source_dir, target_dir)
    if not os.path.exists(os.path.dirname(target_path)):
        try:
            os.makedirs(os.path.dirname(target_path))
        except:
            pass
    cv2.imwrite(target_path, image)
    os.remove(source_path)


def process_tar_file(index, target_dir, resized_dir):
    tar_url = images_base_url.format(index)
    md5_url = md5_base_url.format(index)

    tar_path = os.path.join(*[target_dir, 'tars', os.path.basename(tar_url)])
    md5_path = os.path.join(*[target_dir, 'tars', os.path.basename(md5_url)])

    print('Downloading: ' + tar_path)

    start_time = time.time()

    if not os.path.exists(md5_path):
        urllib.request.urlretrieve(md5_url, md5_path)
    if not os.path.exists(tar_path):
        urllib.request.urlretrieve(tar_url, tar_path)

    print('{}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

    # checksum

    ref_checksum = open(md5_path).readlines()[0].split()[0]
    tar_checksum = md5(tar_path)
    if ref_checksum != tar_checksum:
        print('{}: failed checksum'.format(index))
        return

    # open tar file

    extract_dir = os.path.join(target_dir, 'raw_images')
    tar_file = tarfile.open(tar_path)
    tar_file.extractall(extract_dir)
    tar_file_members = [m.name for m in tar_file.getmembers()]
    tar_file.close()

    # delete tar file

    os.remove(tar_path)
    os.remove(md5_path)

    # resize and move images

    process_func = partial(
        process_image, source_dir=extract_dir, target_dir=resized_dir)
    for file in tqdm(tar_file_members, desc='Files for tar {:03d}'.format(index), mininterval=1.0):
        process_func(file)


def main(target_dir, resized_dir, processes):
    if not os.path.exists(target_dir):
        print('Please create target dir: {}'.format(target_dir))
        return
    if not os.path.exists(os.path.join(target_dir, 'tars')):
        os.mkdir(os.path.join(target_dir, 'tars'))
    if not os.path.exists(os.path.join(target_dir, 'raw_images')):
        os.mkdir(os.path.join(target_dir, 'raw_images'))

    # indexes = list(range(500))
    indexes = list(range(20))
    func = partial(process_tar_file, target_dir=target_dir,
                   resized_dir=resized_dir)
    with Pool(processes) as p:
        for _ in tqdm(p.imap(func, indexes), total=len(indexes), desc='TAR files'):
            pass


if __name__ == "__main__":
    TARGET_DIR = './data/raw/test_stage2'
    RESIZED_DIR = './data/raw/test_stage2_256'

    main(TARGET_DIR, RESIZED_DIR, 10)
