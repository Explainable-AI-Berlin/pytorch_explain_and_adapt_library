import pandas as pd
import numpy as np
from peal.data.datasets import SymbolicDataset
from peal.global_utils import load_yaml_config


class CircleDataset(SymbolicDataset):
    __name__ = "circle"
