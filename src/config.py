IMG_SIZE = (156, 156)
INPUT_SIZE = (128, 128)
INPUT = './data/raw/'
INDEX_PATH = INPUT + 'index.csv'
TRAIN_PATH = INPUT + 'train.csv'
TEST_PATH = INPUT + 'test.csv'
SUBMIT_PATH = INPUT + 'recognition_sample_submission.csv'
TRAIN_IMG_PATH = INPUT + 'train_448/'
TEST_IMG_PATH = INPUT + 'test/'
INDEX_IMG_PATH = INPUT + 'index/'
USE_PRETRAINED = True
PRETRAIN_PATH = './models/gem_seresnet50_4/'
RESET_OPTIM = False

# config
MODEL = 'delf_seresnet50'
BATCH_SIZE_TRAIN = 250
NUM_WORKERS = 16
EPOCHS = 12
PRINT_FREQ = 100
LEARNING_RATE = 1e-4
latent_dim = 1024
DROPOUT_RATE = 0.2
S_TEMPERATURE = 300
N_SELECT = 19
N_UNIQUES = 20
RUN_TTA = True
N_TTA = 4

# non_landmark
PLACES365_PATH = './data/places365/categories_places365_extended.csv'
NON_LANDMARK_PATH = './data/places365/non_landmark.csv'
NON_LANDMARK_IMG_PATH = './data/places365/train/places365_indoor/'
