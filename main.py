from functions.deploy_config import deploy
from functions.data_pipeline import get_datasets
from functions.model_handler import ModelHandler
from tensorflow.keras import Model, layers
from functions import conv_blocks
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='model1', help='options: in developing')
argparser.add_argument('--phase', type=str, default='train', help='options: train, test, vis')
args = argparser.parse_args()

config = deploy(args)

datasets = get_datasets(config)
tr = datasets[0]
val = datasets[1]

num = 0
for x, y in tr:
    num += len(x)
ModelHandler(datasets, config)

