import numpy as np
import pdb
import caffe

from lib import run_net
from lib import score_util
from lib import plot_util

from nets.roads import Road
from nets.kitti import kitti

import os


caffe.set_device(0)
caffe.set_mode_gpu()

#caffe.set_mode_cpu()

net_path = 'nets/stage-cityscapes-fcn8s.prototxt'
weight_path = 'nets/stage-cityscapes-fcn8s.prototxt'

net = caffe.Net(net_path, weight_path, caffe.TEST)

R = Road()
KT = kitti()

n_cl = len(R.classes)

def sm_diff(prev_fts, fts):
    prev_m = prev_fts.argmax(axis=0).copy()
    curr_m = fts.argmax(axis=0).copy()
    diff = np.array(prev_m != curr_m).mean()
    return diff

def adaptive_clockwork(thresh):
    hist = np.zeros((n_cl, n_cl))
    num_frames = 0
    num_update_frames = 0
    for vid in R.list_vids():
        is_first = True
        for f in R.list_frames(vid):
            num_frames += 1 # index the total number of frames
            if is_first: # push the 10 frame lag through the net
                im = R.load_image(vid, f)
                _ = run_net.segrun(net, R.preprocess(im))
#                prev_fts = net.blobs['kitti_score_pool4'].data[0].copy()
                prev_fts = net.blobs['score_pool4'].data[0].copy()
                is_first = False

            # Run to pool4 on current frame
            im = R.load_image(vid, f)	
            run_net.feed_net(net, R.preprocess(im))
#            net.forward(start='conv1_1', end='kitti_score_pool4')
            net.forward(start='conv1_1', end='score_pool4')
#            curr_fts = net.blobs['kitti_score_pool4'].data[0].copy()
            curr_fts = net.blobs['score_pool4'].data[0].copy()

            # Decide whether or not to update to fc7
            d = sm_diff(prev_fts, curr_fts)
            if sm_diff(prev_fts, curr_fts) >= thresh: # push through rest of net
                #net.forward(start='conv5_1', end='kitti_upscore2')
                net.forward(start='conv5_1', end='upscore2')
                #prev_fts = net.blobs['kitti_score_pool4'].data[0].copy()
                prev_fts = net.blobs['score_pool4'].data[0].copy()
                num_update_frames += 1

            # Compute full merge score
            net.forward(start='score_pool4c')
            out = net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)
            KT.palette(out).save(f)
    #        hist += score_util.fast_hist(label.flatten(), out_yt.flatten(), n_cl)

#    acc, cl_acc, mean_iu, fw_iu = score_util.get_scores(hist)
#    print 'Adaptive Clockwork: Threshold', thresh, ' Updated {:d}/{:d} frames ({:2.1f}%)'.format(num_update_frames, num_frames, 100.0*num_update_frames/num_frames)
#    print 'acc\t\t cl acc\t\t mIU\t\t fwIU'
#    print '{:f}\t {:f}\t {:f}\t {:f}\t'.format(100*acc, 100*cl_acc, 100*mean_iu, 100*fw_iu)
#    return acc, cl_acc, mean_iu, fw_iu

adaptive_clockwork(0.1)

