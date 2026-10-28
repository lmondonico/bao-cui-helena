"""
Essential framework imports for regression tasks.
Only universal imports - no defaults or study-specific elements.
"""

import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
import re
import inspect

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Optional