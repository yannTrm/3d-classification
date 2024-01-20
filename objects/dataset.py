# -*- coding: utf-8 -*-
# Import 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import os
import io
import gzip
import random
import numpy as np
import matplotlib.pyplot as plt

import trimesh

from PIL import Image

from torch.utils.data import Dataset

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, shuffle=True):
        """
        Custom dataset for loading 2D images from the specified folder structure.

        Parameters:
        - root_dir (str): The root directory containing subfolders for each class.
        - transform (callable, optional): Optional transform to be applied to the images.
        - shuffle (bool): If True, shuffle the order of images in the dataset.
        """
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.transform = transform
        self.shuffle = shuffle

        self.filepaths, self.labels = self.load_dataset()

        if self.shuffle:
            self.shuffle_dataset()


    def load_dataset(self):
        filepaths = []
        labels = []
        for class_folder in self.classes:
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                for model_folder in os.listdir(class_path):
                    model_path = os.path.join(class_path, model_folder)
                    if os.path.isdir(model_path):
                        for filename in os.listdir(model_path):
                            filepath = os.path.join(model_path, filename)
                            filepaths.append(filepath)
                            labels.append(self.class_to_idx[class_folder])

        return filepaths, labels
    


    def shuffle_dataset(self):
        combined = list(zip(self.filepaths, self.labels))
        random.shuffle(combined)
        self.filepaths[:], self.labels[:] = zip(*combined)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
    
    def plot_image(self, idx):
        """
        Plot the image at the specified index along with its label.

        Parameters:
        - idx (int): Index of the image in the dataset.
        """
        img, label = self.__getitem__(idx)

        # Convert tensor to numpy array
        img_np = img.permute(1, 2, 0).numpy()

        # Plot the image
        plt.imshow(img_np)
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()
