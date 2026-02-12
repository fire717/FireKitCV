import os,argparse
import random
        
from firelib import init_all, FireModel, FireRunner, FireData

from config import cfg




def main(cfg):

    init_all(cfg)

    model = FireModel(cfg)

    data = FireData(cfg)


    train_loader, val_loader = data.get_trainval_dataloader()

    runner = FireRunner(cfg, model)
    runner.train(train_loader, val_loader)


if __name__ == '__main__':
    main(cfg)