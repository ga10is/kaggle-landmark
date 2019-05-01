IMG_SIZE = (256, 256)
INPUT_SIZE = (224, 224)
INPUT = './landmark/data/raw/'
INDEX_PATH = INPUT + 'index.csv'
TRAIN_PATH = INPUT + 'train.csv'
TEST_PATH = INPUT + 'test.csv'
SUBMIT_PATH = INPUT + 'recognition_sample_submission.csv'
TRAIN_IMG_PATH = INPUT + 'train_448/'
TEST_IMG_PATH = INPUT + 'test/'
INDEX_IMG_PATH = INPUT + 'index/'

# config
BATCH_SIZE_TRAIN = 64
NUM_WORKERS = 8
EPOCHS = 12
PRINT_FREQ = 100
latent_dim = 512
