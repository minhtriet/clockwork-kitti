import os

import pdb

import sys

import numpy as np
#import matplotlib.pyplot as plt

from scipy import misc

import caffe
from nets.roads import Road

from datasets.cityscapes import cityscapes

from lib import run_net
from lib import score_util
from lib import plot_util

import time

#plt.rcParams['image.cmap'] = 'gray'
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['figure.figsize'] = (12, 12)

caffe.set_device(0)
caffe.set_mode_gpu()

KT = Road()

n_cl = 19

net = caffe.Net('nets/stage-cityscapes-fcn8s.prototxt', 
                'snapshot_cc_iter_100000.caffemodel',
                caffe.TEST)

dir_path = os.path.dirname(os.path.realpath(__file__))

def sm_diff(prev_scores, scores):
    prev_seg = prev_scores.argmax(axis=0).astype(np.uint8).copy()
    curr_seg = scores.argmax(axis=0).astype(np.uint8).copy()
    diff = np.array(prev_seg != curr_seg).mean()
    return diff

def adaptive_clockwork(thresh):
    hist = np.zeros((n_cl, n_cl))
    num_frames = 0
    num_update_frames = 0
    vid = KT.list_vids()[1]
    tic = time.clock()
    is_first = True
    for f in KT.list_frames(vid):
        num_frames += 1 # index the total number of frames
        if is_first: # push the 10 frame lag through the net
            im = KT.load_image(vid, f)
            _ = run_net.segrun(net, KT.preprocess(im))
            mintoc = time.clock()
            prev_fts = net.blobs['score_pool4'].data[0].copy()
            is_first = False
            print "!!!!!!!!! %f" % (tic-mintoc)

        # Run to pool4 on current frame
        im = KT.load_image(vid, f)	
        run_net.feed_net(net, KT.preprocess(im))
        net.forward(start='conv1_1', end='score_pool4')
        curr_fts = net.blobs['score_pool4'].data[0].copy()

        # Decide whether or not to update to fc7
        d = sm_diff(prev_fts, curr_fts)
        if sm_diff(prev_fts, curr_fts) >= thresh: # push through rest of net
            net.forward(start='conv5_1', end='upscore2')
            prev_fts = net.blobs['score_pool4'].data[0].copy()
            num_update_frames += 1

        # Compute full merge score
        net.forward(start='score_pool4c')
        out = net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)
        misc.imsave(f, KT.palette(out))
    toc = time.clock()
    print 'Adaptive Clockwork: Threshold', thresh, ' Updated {:d}/{:d} frames ({:2.1f}%)'.format(num_update_frames, num_frames, 100.0*num_update_frames/num_frames)
    print tic - toc

adaptive_clockwork(0.05)

