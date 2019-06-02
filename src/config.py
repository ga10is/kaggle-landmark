IMG_SIZE = (145, 145)
INPUT_SIZE = (128, 128)
INPUT = './data/raw/'
INDEX_PATH = INPUT + 'index.csv'
TRAIN_PATH = INPUT + 'train.csv'
TEST_PATH = INPUT + 'test_stage2.csv'
SUBMIT_PATH = INPUT + 'test_stage2.csv'  # 'recognition_sample_submission.csv'
TRAIN_IMG_PATH = INPUT + 'train_448/'
TEST_IMG_PATH = INPUT + 'test_stage2_256/'
INDEX_IMG_PATH = INPUT + 'index/'
USE_PRETRAINED = True
PRETRAIN_PATH = './models/delf_seresnet50_finetune_7/'
RESET_OPTIM = False

# config
MODEL = 'delf_seresnet50_finetune'
BATCH_SIZE_TRAIN = 200
NUM_WORKERS = 16
EPOCHS = 12
PRINT_FREQ = 100
LEARNING_RATE = 1e-4
latent_dim = 1024
DROPOUT_RATE = 0.2
S_TEMPERATURE = 200
N_SELECT = 5
N_UNIQUES = 6
RUN_TTA = True
N_TTA = 4

# non_landmark
PLACES365_PATH = './data/places365/categories_places365_extended.csv'
NON_LANDMARK_PATH = './data/places365/non_landmark.csv'
NON_LANDMARK_IMG_PATH = './data/places365/train/places365_indoor/'
