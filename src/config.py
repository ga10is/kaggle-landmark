IMG_SIZE = (256, 256)
INPUT_SIZE = (224, 224)
INPUT = './data/raw/'
INDEX_PATH = INPUT + 'index.csv'
TRAIN_PATH = INPUT + 'train.csv'
TEST_PATH = INPUT + 'test.csv'
SUBMIT_PATH = INPUT + 'recognition_sample_submission.csv'
TRAIN_IMG_PATH = INPUT + 'train_448/'
TEST_IMG_PATH = INPUT + 'test/'
INDEX_IMG_PATH = INPUT + 'index/'
USE_PRETRAINED = True
PRETRAIN_PATH = './models/resnet18_5/'

# config
BATCH_SIZE_TRAIN = 300
NUM_WORKERS = 12
EPOCHS = 12
PRINT_FREQ = 100
LEARNING_RATE = 1e-4
latent_dim = 512
DROPOUT_RATE = 0.2
S_TEMPERATURE = 100
N_SELECT = 2
