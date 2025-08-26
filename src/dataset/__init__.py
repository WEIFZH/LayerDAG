from .layer_dag import *
from .general import DAGDataset
from .tpu_tile import get_tpu_tile
from .cd_syn import get_cd_syn
def load_dataset(dataset_name):
    if dataset_name == 'tpu_tile':
        return get_tpu_tile()
    elif dataset_name == 'cd_syn':
        return get_cd_syn()
    else:
        return NotImplementedError
