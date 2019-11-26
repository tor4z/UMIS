from .hela import HelaDataset
from .vesselNN import vesselNN

datasets = {
    'hela': HelaDataset,
    'vesslNN': vesslNN
}


def get_dataset(opt):
    dataset_cls = datasets[opt.dataset]
    return dataset_cls(opt)
