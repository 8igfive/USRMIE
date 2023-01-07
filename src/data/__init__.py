from . import wsd_dataset
from . import qa_dataset

DATASETS = {
    'wsd': wsd_dataset.Dataset, # wsd_dataset.build_dataset
    'qa': qa_dataset.Dataset
}