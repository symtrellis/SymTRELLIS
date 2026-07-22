from typing import Callable, Dict, Type

from . import abo, local, objaversexl, toys4k
from .base import DatasetWorkspace

data_dict: Dict[str, Type[DatasetWorkspace]] = {
    "local": local.LocalDataset,
    "ABO": abo.ABODataset,
    "ObjaverseXL": objaversexl.ObjaverseXLDataset,
    "Toys4K": toys4k.Toys4KDataset,
}

args_dict: Dict[str, Callable] = {
    "local": local.add_args,
    "ABO": abo.add_args,
    "ObjaverseXL": objaversexl.add_args,
    "Toys4K": toys4k.add_args,
}
