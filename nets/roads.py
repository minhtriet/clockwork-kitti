import os
import sys
import glob
import numpy as np

import PIL
from PIL import Image

import pdb

class Road:
    def __init__(self, data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        self.dir = os.path.join(data_path, 'datasets')
        self.mean = (104.00698793, 116.66876762, 122.67891434) # imagenet mean
        self.MAX_DIM = 500.0  # match PASCAL VOC training data
        sys.path.insert(0, '{}/x/cityscapes/scripts/helpers/'.format(os.path.dirname(self.dir)))
        labels = __import__('labels')
        labels_and_ids = [(l.name, l.trainId) for l in labels.labels if l.trainId >= 0 and l.trainId < 255]
        self.classes = [l[0] for l in sorted(labels_and_ids, key=lambda x: x[1])]  # classes in ID order == network output order
        self.id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs
        self.id2color = {label.id: label.color for label in labels.labels}  # dictionary mapping train IDs to colors as 3-tuples
 

    def list_vids(self):
        scenes = [os.path.basename(f) for f in glob.glob('{}/road/2011_09_26/*'.format(self.dir))]
        return scenes 

    def list_frames(self, vid):
        f = [os.path.basename(f) for f in glob.glob('{}/road/2011_09_26/{}/image_03/data/*'.format(self.dir, vid))]
        f.sort()
        return f

    def load_image(self, vid, idx):
        im = Image.open('{}/road/2011_09_26/{}/image_03/data/{}'.format(self.dir, vid, idx))
        im = self.resize(im, False)
        return im

    def resize(self, im, label=False):
        dims = np.array(im).shape
        if len(dims) > 2:
            dims = dims[:-1]
        max_val, max_idx = np.max(dims), np.argmax(dims)
        scale = self.MAX_DIM / max_val
        new_height, new_width = int(dims[0]*scale), int(dims[1]*scale)
        if label:
            im = im.resize((new_width, new_height), resample=PIL.Image.NEAREST)
        else:
            im = im.resize((new_width, new_height), resample=PIL.Image.BILINEAR)
        return im

    def preprocess(self, im):
        """
        Preprocess loaded image (by load_image) for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        in_ = Image.new("RGB", im.size)
        in_.paste(im)
        in_ = np.array(in_, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def palette(self, label):
        '''
        Map trainIds to colors as specified in labels.py
        '''
        if label.ndim == 3:
            label= label[0]
        color = np.empty((label.shape[0], label.shape[1], 3))
        for k, v in self.id2color.iteritems():
            color[label == k, :] = v
        return color
