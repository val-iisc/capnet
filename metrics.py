'''
Code for computing chamfer and emd metrics for the baseline and projection 
trained models.
Use run_metrics.sh to run the code
'''

import os, sys
sys.path.append('./src')
sys.path.append('./src/utils_chamfer')
import json
import argparse
import cv2
import glob
import numpy as np
import random
import re
import scipy
import tensorflow as tf
import tflearn
import time
from itertools import product
from scipy import misc
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename
from itertools import product
import csv
from tqdm import tqdm
import pdb

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)

from net import recon_net_large as recon_net
import tf_nndistance
from tf_auctionmatch import auction_match
from blend_background import blendBg
from shapenet_taxonomy import shapenet_id_to_category, shapenet_category_to_id
from utils_metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, 
        help='Name of Experiment Prefixed with index')
parser.add_argument('--setup', type=str, required=True, 
        help='Experimental setup : ["baseline", "projection"]')
parser.add_argument('--gpu', type=str, required=True, 
        help='GPU to use - in machines with more than one GPU')
parser.add_argument('--category', type=str, required=True, 
        help='Category to visualize from : ["airplane", "car", "chair"]')
parser.add_argument('--eval_set', type=str, required=True, 
        help='set to compute metrics on : ["train", "val", "test"]')
parser.add_argument('--snapshot', type=str, required=True, 
        help='Load snapshot : ["latest" ,"best"]')
parser.add_argument('--batch_size', type=int, default=10, 
        help='Batch Size during evaluation. Make sure to set a value that\
        perfectly divides the total number of samples.')
parser.add_argument('--natural', action='store_true', 
        help='Set to true if images should be overlayed on natural background')
parser.add_argument('--rotate', action='store_true', 
        help='Supply this parameter to rotate the point cloud.\
        For canonical, ignore.')
parser.add_argument('--scale_outlier', action='store_true', 
        help='scaling function to use if outlier points are present')
parser.add_argument('--IMG_H', type=int, default=64, 
        help='input image height')
parser.add_argument('--IMG_W', type=int, default=64, 
        help='input image width')
parser.add_argument('--N_VIEWS', type=int, default=10, 
        help='Number of views from which pcl is projected')
parser.add_argument('--OUTPUT_PCL_SIZE', type=int, default=1024, 
        help='Number of points in predicted PCL')
parser.add_argument('--bottleneck', type=int, default=128, 
        help='dimension of encoder output')
# visualize
parser.add_argument('--ballradius', type=int, default=3, 
        help='Radius of points in visualization')
parser.add_argument('--visualize', action='store_true', 
        help='visualize generated point clouds')
parser.add_argument('--save_screenshots', action='store_true', 
        help='save screenshots')
parser.add_argument('--save_gifs', action='store_true', 
        help='save gifs')
# misc
parser.add_argument('--tqdm', action='store_true', 
        help='view progress bar')

args = parser.parse_args()

print '-='*50
print args
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

data_dir = 'data/ShapeNet_rendered'
data_dir_pcl = 'data/ShapeNet_v1_pcl'
data_dir_224 = 'data/ShapeNet_rendered_res_224'
sun_dir = 'data/sun2012pascalformat'

random.seed(1024)

BATCH_SIZE = args.batch_size

PNG_FILES = ['render_0.png', 'render_1.png', 'render_2.png', 'render_3.png', 'render_4.png', 'render_5.png', 'render_6.png', 'render_7.png', 'render_8.png', 'render_9.png']

if args.natural:
    category_id = shapenet_category_to_id[args.category]
    bgImgsList = listdir(sun_dir)
    bgImgsList = [join(sun_dir, img_path) for img_path in bgImgsList]


if __name__=='__main__':
    # Snapshot Folder Location
    if args.snapshot == 'best':
            snapshot_folder = join(args.exp, 'best')
    elif args.snapshot == 'latest':
            snapshot_folder = join(args.exp, 'snapshots')

    # use case
    if args.visualize:
        import show3d_balls
    elif args.save_screenshots or args.save_gifs:
        import show3d_balls
        screenshot_dir = join(args.exp, 'screenshots')
        create_folder(screenshot_dir)
    else:
        if args.exp == '':
            print 'exp name is empty! Check code.'
        csv_path = join(args.exp, '%s_%s.csv'%(args.eval_set, args.snapshot))
        csv_path = os.path.abspath(csv_path)
        with open(csv_path, 'w') as f:
            f.write('Id; Chamfer; Fwd; Bwd; Emd\n')

    # placeholders
    img_inp = tf.placeholder(tf.float32, shape=(None,args.IMG_H,args.IMG_W,3), 
            name='img_inp')
    gt_pcl = tf.placeholder(tf.float32, shape=(None,args.OUTPUT_PCL_SIZE,3), 
            name='pcl_gt')
    pred_pcl = tf.placeholder(tf.float32, shape=(None,args.OUTPUT_PCL_SIZE,3), 
            name='pcl_pred')

    # Build graph
    with tf.variable_scope('recon_net'):
        pred_pcl = recon_net(img_inp, args)

    # metrics
    if args.scale_outlier:
        gt_pcl_scaled, pred_pcl_scaled = scale_outlier(gt_pcl, pred_pcl)
    else:
        gt_pcl_scaled, pred_pcl_scaled = scale(gt_pcl, pred_pcl)

    dists_forward, dists_backward, chamfer_distance, emd = get_metrics(gt_pcl_scaled, pred_pcl_scaled, args)
    
    # GPU configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Run session
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        # saver to load previous checkpoint
        saver = tf.train.Saver()
        load_previous_checkpoint(snapshot_folder, saver, sess)
        
        tflearn.is_training(False, session=sess)

        # data
        if args.natural:
            models, indices  = get_drc_models(data_dir_224, args)
        else:
            models, indices = get_drc_models(data_dir, args)

        if args.visualize or args.save_screenshots or args.save_gifs:
            random.shuffle(indices)
        batches = len(indices) // BATCH_SIZE

        print('computing metrics for %d samples...'%len(indices))
        iters = range(batches)
        if args.tqdm:
                iters = tqdm(iters)

        for cnt in iters:
            # load batch
            batch_ip, batch_gt = fetch_batch(models, indices, cnt, BATCH_SIZE,
                    data_dir_pcl, args)
            fids = fetch_batch_paths(models, indices, cnt, BATCH_SIZE)
            _pred_pcl = sess.run(pred_pcl, feed_dict={img_inp:batch_ip, gt_pcl:batch_gt})

            # rotate pred pcl to canonical frame
            if args.rotate:
                _pred_temp = []
                for b in xrange(BATCH_SIZE):
                    ind = indices[cnt*BATCH_SIZE+b]
                    model_path = models[ind[0]]
                    angles_path = join(model_path, 'view.txt')
                    with open(angles_path, 'r') as fp:
                        angles = [item.split('\n')[0] for item in fp.readlines()]
                        angle = angles[ind[1]]
                        xangle = np.pi / 180. * float(angle.split(' ')[0])
                        yangle = np.pi / 180. * float(angle.split(' ')[1])
                    pred = np_rotate(_pred_pcl[b], xangle=-xangle, 
                            yangle=yangle, inverse=True)
                    _pred_temp.append(pred)
                _pred_pcl = np.array(_pred_temp, dtype=np.float32)

            # metrics
            # C,F,B,E are all arrays of dimension (BATCH_SIZE,)
            if args.setup == 'projection':# or args.rotate:
                _pred_pcl = rotate(rotate(_pred_pcl,0,90).eval(),90,0).eval()

            if args.rotate:
                _pred_pcl = rotate(rotate(_pred_pcl,0,90).eval(),90,0).eval()

            _gt_scaled, _pred_scaled = sess.run([gt_pcl_scaled, pred_pcl_scaled], 
                feed_dict={gt_pcl:batch_gt, pred_pcl:_pred_pcl})
            C,F,B,E = sess.run([chamfer_distance, dists_forward, 
                dists_backward, emd], 
                feed_dict={gt_pcl_scaled:_gt_scaled, 
                    pred_pcl_scaled:_pred_scaled})

            # visualize
            if args.visualize:
                # Rotate point clouds to align axes
                pr = rotate(_pred_scaled,-90,-90).eval()
                gt = rotate(_gt_scaled,-90,-90).eval()
                for b in xrange(BATCH_SIZE):
                    print 'Model:{} C:{:.6f} F:{:.6f} B:{:.6f} E:{:.6f}'.format(fids[b],C[b],F[b],B[b],E[b])
                    cv2.imshow('', batch_ip[b])
                    show3d_balls.showpoints(pr[b], ballradius=3)
                    show3d_balls.showpoints(gt[b], ballradius=3)
                    saveBool = show3d_balls.showtwopoints(gt[b], pr[b],
                            ballradius=args.ballradius)

            # screenshots and gifs
            elif args.save_screenshots or args.save_gifs:
                # Rotate point clouds to align axes
                pr = rotate(_pred_scaled,-90,-90).eval()
                gt = rotate(_gt_scaled,-90,-90).eval()
                for b in xrange(BATCH_SIZE):
                    save_screenshots(gt[b], pr[b], batch_ip[b], 
                           screenshot_dir, fids[b], args.eval_set, args)
                print 'done'

            # save metrics to csv
            else:
                if np.isnan(C).any() or np.isnan(E).any():
                    print fids
                    print C
                    print E
                else:
                    with open(csv_path, 'a') as f:
                        for b in xrange(BATCH_SIZE):
                            f.write('{};{:.6f};{:.6f};{:.6f};{:.6f}\n'.format(fids[b],C[b],F[b],B[b],E[b]))

        # get avg metrics
        C_avg,F_avg,B_avg,E_avg = get_averages(csv_path)
        print 'Final Metrics:  Chamfer  Forward  Backward  EMD'.format(C_avg,F_avg,B_avg,E_avg)
        print '{:.6f};  {:.6f};  {:.6f};  {:.6f}'.format(C_avg,F_avg,B_avg,E_avg)

