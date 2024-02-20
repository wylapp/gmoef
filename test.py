from attrdict import AttrDict
import yaml
from src import dataset
import time


def load_config(config_file):
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error while loading config file: {e}")

config = AttrDict(
    yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
       

dm = dataset.M4DataModule(config)
dm.prepare_data()
dm.setup(stage='fit')

for train in dm.train_dataloader():
    print(train)
    time.sleep(3)