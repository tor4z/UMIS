from torch.utils.data import DataLoader

from utils.config import Config
from utils.saver import Saver
from utils.summaries import Summary
from models.trainner import Trainner
from data import SingleFoldValid

from models.segnet import SegNet2D

CFG_FILE = 'cfg.yaml'


def main(opt):
    saver = Saver(opt)
    summary = Summary(opt)
    sfv =  SingleFoldValid(opt)
    trainner = Trainner(opt)
    for train_set, val_set in sfv.gen_dataset():
        train_dataloader = DataLoader(train_set, 
                            batch_size=opt.batch_size,
                            shuffle=opt.shuffle,
                            num_workers=opt.num_workers)
        val_dataloader = DataLoader(val_set)
    trainner.run(train_dataloader, val_dataloader)


if if __name__ == "__main__":
    opt = Config(CFG_FILE)
    main(opt)