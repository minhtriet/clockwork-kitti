#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 23:16:49 2016

@author: minhtriet
"""

import caffe

import numpy as np
from PIL import Image

import os
from nets.roads import Road
from nets.pascal_voc import pascal

import pdb

R = Road()
PV = pascal('{}/datasets/VOCdevkit/VOC2012'.format(os.path.dirname(os.path.abspath(__file__))))

import random

class RoadSegDataLayer(caffe.Layer):

    """
    Load (input image, label image) pairs from Kitti
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.dir = R.dir
        self.mean = np.array(params['mean'])
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.indices = R.list_frames(R.list_vids()[0])
        self.idx = 0


    def reshape(self, bottom, top):
        # load image
        im = R.load_image(R.list_vids()[0], self.indices[self.idx])
        self.data = R.preprocess(im)
        self.label = self.data  # hack

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        self.idx += 1
        if self.idx == len(self.indices):
            self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass

