# -*- coding: utf-8 -*-
# Import 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import os
import json
import numpy as np

import torch

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def split_dataset_1(dataset, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - dataset (Dataset): The dataset to be split.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Seed for random state for reproducibility.

    Returns:
    - train_dataset (Dataset): Training set.
    - test_dataset (Dataset): Testing set.
    """
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=dataset.labels,  # Ensures proportional class distribution in train and test sets
        random_state=random_state
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, test_dataset

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
