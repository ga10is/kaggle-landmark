IMG_SIZE = (145, 145)
INPUT_SIZE = (128, 128)
INPUT = './data/raw/'
TRAIN_PATH = INPUT + 'train.csv'
TEST_PATH = INPUT + 'test_stage2.csv'
SUBMIT_PATH = INPUT + 'test_stage2.csv'
TRAIN_IMG_PATH = INPUT + 'train_448/'
TEST_IMG_PATH = INPUT + 'test_stage2_256/'
INDEX_IMG_PATH = INPUT + 'index/'
USE_PRETRAINED = False
PRETRAIN_PATH = './models/delf_seresnet50_finetune_9/'
RESET_OPTIM = False

# config
MODEL = 'delf_seresnet50_finetune'
BATCH_SIZE_TRAIN = 200
NUM_WORKERS = 16
EPOCHS = 15
PRINT_FREQ = 100
LEARNING_RATE = 5e-5
latent_dim = 1024
DROPOUT_RATE = 0.3
# Temperature for ArcFace loss
S_TEMPERATURE = 200
# The number of images sampled for each class per epoch.
N_SELECT = 5
# Threshold for selecting classes. Select classes that has images above the threshold.
N_UNIQUES = 6
RUN_TTA = True
N_TTA = 4

# non_landmark
PLACES365_PATH = './data/places365/categories_places365_extended.csv'
NON_LANDMARK_PATH = './data/places365/non_landmark.csv'
NON_LANDMARK_IMG_PATH = './data/places365/train/places365_indoor/'
