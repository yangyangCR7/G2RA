from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []
        self.image_ids = [] # Added to store image IDs for verification

        # Backup storage for reset
        self.ys_orig = None
        self.im_paths_orig = None
        self.I_orig = None
        self.image_ids_orig = None

    def nb_classes(self):
        if not set(self.ys).issubset(self.classes):
            extra_labels = set(self.ys) - self.classes
            print(f"Warning: Found labels {extra_labels} not in self.classes!")
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target = self.ys[index]

        return im, target

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        # Initialize backup if it's the first time modifying
        if self.ys_orig is None:
            self.ys_orig = self.ys[:]
            self.im_paths_orig = self.im_paths[:]
            self.I_orig = self.I[:]
            if hasattr(self, 'image_ids') and len(self.image_ids) > 0:
                self.image_ids_orig = self.image_ids[:]
            else:
                self.image_ids_orig = []

        # If I is None, reset to original full dataset
        if I is None:
            self.ys = self.ys_orig[:]
            self.im_paths = self.im_paths_orig[:]
            self.I = self.I_orig[:]
            if self.image_ids_orig:
                self.image_ids = self.image_ids_orig[:]
            return

        # Slice from the *ORIGINAL* data to ensure indices I are always relative to full set
        self.ys = [self.ys_orig[i] for i in I]
        self.I = [self.I_orig[i] for i in I]
        self.im_paths = [self.im_paths_orig[i] for i in I]
        if self.image_ids_orig:
            self.image_ids = [self.image_ids_orig[i] for i in I]