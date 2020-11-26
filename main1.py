from functions.deploy_config import deploy
from functions.data_pipeline import get_datasets
from functions.model_handler import ModelHandler
from tensorflow.keras import Model, layers
from functions import conv_blocks
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='model1', help='options: in developing')
argparser.add_argument('--phase', type=str, default='test', help='options: train, test')
args = argparser.parse_args()

config = deploy(args)
ModelHandler(datasets, config)
