import os

DATA_DIR = os.path.join(os.getcwd(), 'data')
DATA_PATH = os.path.join(DATA_DIR, 'train.csv')

MODELS_DIR = os.path.join(os.getcwd(), 'models')
MODELS_PATH = os.path.join(MODELS_DIR, 'ANN.keras')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')
