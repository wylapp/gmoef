import argparse
import yaml
from attrdict import AttrDict
from src.model import ClicheTeller
from src.dataset import M4DataModule
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy
import os
from src import utils
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint


def main(args):
    """main entry point for the whole lightning model.

    Args:
        args: configures
    """
    config = AttrDict(
            yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))

    for k, v in vars(args).items():
        setattr(config, k, v)
    # * setup the logger
    
    seed_everything(seed=config.seed, workers=True)

    # * setup checkpoint saving behaviors
    ckpt = ModelCheckpoint(save_top_k=5, save_on_train_epoch_end=True, monitor='evl/ACC',
                           mode='max')
    if 'train' in config.mode:
        # * prepare the trainer
        tb_logger = pl_loggers.TensorBoardLogger('log')
        # wandb_logger = pl_loggers.WandbLogger(offline=True)
        loggers = tb_logger
        trainer = L.Trainer(accelerator='gpu', 
                            devices=[0,1],
                            strategy=DDPStrategy(find_unused_parameters=True), 
                            precision='bf16-mixed',
                            max_epochs=config.epoch,
                            sync_batchnorm=True,
                            logger=loggers,
                            val_check_interval=0.2,
                            callbacks=[ckpt])
        
        # * prepare data and model
        dm = M4DataModule(config)
        cliche = ClicheTeller(config=config)

        trainer.fit(cliche, datamodule=dm)
        # trainer.validate(cliche, datamodule=dm)
        # trainer.test(cliche, ckpt_path='/13804111422/task8/log/lightning_logs/version_5/checkpoints/epoch=2-step=12933.ckpt', datamodule=dm)
        trainer.test(cliche, datamodule=dm)
    elif 'predict' in config.mode:
        trainer = L.Trainer(accelerator='gpu', 
                            devices=[0],
                            precision='bf16-mixed',
                            inference_mode=True)
        
        # * prepare data and model
        dm = M4DataModule(config)
        cliche = ClicheTeller(config=config)
        res = trainer.predict(cliche, ckpt_path='/13804111422/task8/log/lightning_logs/version_8/checkpoints/epoch=0-step=1497.ckpt', datamodule=dm)
        # utils.dump_final_result(res)
    elif 'test' in config.mode:
        trainer = L.Trainer(accelerator='gpu', 
                            devices=[0],
                            precision='bf16-mixed',
                            inference_mode=True)
        
        # * prepare data and model
        dm = M4DataModule(config)
        cliche = ClicheTeller(config=config)
        trainer.test(cliche, ckpt_path='/13804111422/task8/log/lightning_logs/version_9/checkpoints/epoch=0-step=1872.ckpt', datamodule=dm)
        # utils.dump_final_result(res)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='test',
                        choices=['train_test', 'train', 'test', 'predict'])
    parser.add_argument("--pooler_type",
                        type=str,
                        default='avg',
                        choices=['avg', 'attention', 'cls', 'avg_first_last', 'avg_top2', 'gate'])
    parser.add_argument("--decision_make",
                        type=str,
                        default='cross',
                        choices=['avg', 'cross', 'concat', 'trade-off'])
    args = parser.parse_args()
    main(args=args)