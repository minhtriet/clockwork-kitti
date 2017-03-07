import os
import sys
import glob
import numpy as np

import PIL
from PIL import Image

import pdb

class kitti:
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
 
    def list_vids(self, split):
        scenes = [os.path.basename(f) for f in glob.glob('{}/data_road/{}/image_2/*'.format(self.dir, split))]
        return scenes 

    def load_image(self, split, idx):
        print "Load image {}".format(idx)
        im = Image.open('{}/data_road/{}/image_2/{}'.format(self.dir, split, idx))
        im = self.resize(im, False)
        return im

    def change_color(self, im, origin_color, new_color):
        im = im.convert('RGBA')

        data = np.array(im)   # "data" is a height x width x 4 numpy array
        red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

        # Replace white with red... (leaves alpha values alone...)
        areas = (red == origin_color[0]) & (blue == origin_color[1]) & (green == origin_color[2])
        data[..., :-1][areas.T] = new_color # Transpose back needed

        im2 = Image.fromarray(data).convert('RGB')
        return im2 


    def load_label(self, split, idx):
        idx = idx.replace("umm_","umm_road_")
        idx = idx.replace("um_","um_lane_")
        idx = idx.replace("uu_","uu_road_")
        im = Image.open('{}/data_road/{}/gt_image_2/{}'.format(self.dir, split, idx))
        # change color to fit new layer
        im = self.resize(im, True)
        im = self.change_color(im, (255, 0, 0), (0, 0, 0))
        im = np.array(im)
        # im = self.change_color(im, (255, 255, 0), (128,64,128))   # road color in cityscape
        label = np.zeros((im.shape[0], im.shape[1]))
        mask = (im==[0,0,0]).all(axis=2)
        label[mask == False] = 7
        label = label[np.newaxis, ...]
        return label

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

    def kt_palette(self, label_im):
        '''
        Transfer the VOC color palette to an output mask
        '''
        if label_im.ndim == 3:
            label_im = label[0]
        color = np.empty((label_im.shape[0], label_im.shape[1], 3))
        for k in np.unique(label_im):
            color[label_im == k, :] = self.kt_palette[k]
        return color

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
