import argparse
from torch.utils.data import DataLoader

from utils.config import Config
from utils.saver import Saver
from utils.summaries import Summary
from models.trainner import Trainner
from data import SingleFoldValid

from models.segnet import SegNet2D


def main(opt):
    saver = Saver(opt)
    summary = Summary(opt)
    sfv =  SingleFoldValid(opt)
    trainner = Trainner(opt, summary, saver)
    trainner.setup_models()
    for fold, (train_set, val_set) in enumerate(sfv.gen_dataset()):
        print('==== Training fold {} ===='.format(fold + 1))

        train_dataloader = DataLoader(train_set, 
                            batch_size=opt.batch_size,
                            shuffle=opt.shuffle,
                            num_workers=opt.num_workers)
        val_dataloader = DataLoader(val_set)
        trainner.run(train_dataloader, val_dataloader)
        break
    print('Finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('UMIS')
    parser.add_argument('--cfg', type=str, help='config file', default='cfg.yaml')
    args = parser.parse_args()

    opt = Config(args.cfg)
    main(opt)