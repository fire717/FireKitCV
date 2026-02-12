import os,argparse
import random
        
from firelib import init_all, FireModel, FireRunner, FireData

from config import cfg




def main(cfg):


    init_all(cfg)


    model = FireModel(cfg)
    
    data = FireData(cfg)

    _, val_loader = data.get_trainval_dataloader()


    runner = FireRunner(cfg, model)


    runner.load_model(cfg['model_path'])


    runner.evaluate(val_loader)



if __name__ == '__main__':
    main(cfg)