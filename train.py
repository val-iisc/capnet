'''
Main code to train the projection based reconstruction network
Use run_train.sh to run the code, modify arguments as needed
'''

import os, sys
sys.path.append('./src')
sys.path.append('./src/utils_chamfer')
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename
import json
import argparse
import numpy as np
import random
import re
import tensorflow as tf
import time
import scipy.misc as sc
import pprint

from utils import *
from net import recon_net_large as recon_net

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)

import tf_nndistance
from proj_codes import cont_proj, perspective_transform, world2cam
from shapenet_taxonomy import shapenet_id_to_category, shapenet_category_to_id
from get_losses import *

parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str, required=True, 
        help='Name of Experiment')
parser.add_argument('--gpu', type=str, required=True, 
        help='GPU id to use')
parser.add_argument('--category', type=str, required=True, 
        help='Category to train: ["airplane", "car", "chair", "all"]')
parser.add_argument('--batch_size', type=int, default=1, 
        help='Batch Size during training')
parser.add_argument('--lr', type=float, default=0.00005, 
        help='Learning Rate')
parser.add_argument('--bottleneck', type=int, default=128, 
        help='encoder output dimension')
parser.add_argument('--N_VIEWS', type=int, default=4, 
        help='Number of projections for loss calculation')
parser.add_argument('--SIGMA_SQ', type=float, default=0.5, 
        help='variance of gaussian in projection formula')
parser.add_argument('--ONLY_BCE', action='store_true',  
        help='Use only bce loss, no affinity loss')
parser.add_argument('--CORR', action='store_true', 
        help='use single projection from corresponding view of input image')
parser.add_argument('--grid_h', type=int, default=64, 
        help='projection grid height')
parser.add_argument('--grid_w', type=int, default=64, 
        help='projection grid width')
parser.add_argument('--max_epoch', type=int, default=100, 
        help='Maximum number of training epochs')
parser.add_argument('--OUTPUT_PCL_SIZE', type=int, default=1024, 
        help='Number of points in predicted pcl')
parser.add_argument('--IMG_H', type=int, default=64, 
        help='Input image height')
parser.add_argument('--IMG_W', type=int, default=64, 
        help='Input image width')
parser.add_argument('--print_n', type=int, default=100, 
        help='Print loss every print_n iters')
parser.add_argument('--save_n', type=int, default=1000, 
        help='Save images every save_n iters')
parser.add_argument('--save_model_n', type=int, default=1000, 
        help='Save checkpoint (network weights) every save_model_n iters')

parser.add_argument('--lambda_bce', type=float, default=1., 
        help='Weight for bce loss')
parser.add_argument('--lambda_aff_fwd', type=float, default=1., 
        help='Weight for forward affinity loss')
parser.add_argument('--lambda_aff_bwd', type=float, default=1., 
        help='Weight for backward affinity loss')
parser.add_argument('--lambda_reg', type=float, default=0., 
        help='Weight for regularization loss')


args = parser.parse_args()

print '-='*50
print args
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

data_dir = 'data/ShapeNet_rendered'
data_dir_pcl = 'data/ShapeNet_v1_pcl'

random.seed(1024)

BATCH_SIZE = args.batch_size
        

if __name__=='__main__':

    if os.path.exists(args.exp):
        resp = raw_input('Directory exists. Continue(y/n) ? ')
        if resp != 'y' and resp != 'yes':
            sys.exit('Directory exists! Rename experiment')

    # Create a folder for experiment and copy the training file
    exp_dir = join(BASE_DIR, args.exp)
    create_folder([exp_dir])
    filename = basename(__file__)
    os.system('cp %s %s'%(filename, exp_dir))

    # Define Log Directories
    snapshot_folder = join(exp_dir, 'snapshots')
    logs_folder = join(exp_dir, 'logs')
    log_file = join(exp_dir, 'logs.txt')
    proj_images_folder = join(exp_dir, 'log_proj_images')
    proj_pcl_folder = join(exp_dir, 'log_proj_pcl')

    # Create log directories
    create_folder([snapshot_folder, logs_folder, proj_images_folder, 
        proj_pcl_folder])

    args_file = join(logs_folder, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    # Load training data
    all_categs = [item for item in args.category.split(' ')]
    category = [shapenet_category_to_id[item] for item in all_categs]
    train_models, val_models, train_pair_indices, val_pair_indices = get_shapenet_drc_models(data_dir, category) 
    train_models_pcl, _, train_pair_indices_pcl,_ = get_shapenet_drc_models(data_dir_pcl, category)
    batches = len(train_pair_indices) / args.batch_size

    # Create placeholders
    img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, args.IMG_H, args.IMG_W, 3), 
            name='img_inp') 
    proj_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, args.N_VIEWS, 
            args.grid_h, args.grid_w), name='proj_gt')
    pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1024, 3),
                name='pcl_gt_1K')
    view_x = tf.placeholder(tf.float32, shape=(BATCH_SIZE,args.N_VIEWS),
            name='view_x')
    view_y = tf.placeholder(tf.float32, shape=(BATCH_SIZE,args.N_VIEWS),
            name='view_y')

    # Tensorboard summary placeholders
    train_loss_summ = []
    loss_names = ['Loss_total', 'Loss_bce', 'Loss_Aff_fwd', 'Loss_Aff_bwd']
    for idx, name in enumerate(loss_names):
        train_loss_summ.append(tf.placeholder(tf.float32, shape=(), name=name))

    # Build graph
    with tf.variable_scope('recon_net'):
            pcl_out = recon_net(img_inp, args)

    pcl_out_rot = {}; proj_pred = {}; pcl_out_persp = {}; loss = 0.;
    loss_bce = {}; fwd = {}; bwd = {}; loss_fwd = {}; loss_bwd = {};
    grid_dist_tensor = grid_dist(args.grid_h, args.grid_w)

    # Projection and loss definition from N_VIEWS viewpoints
    for idx in range(0,args.N_VIEWS):
        # World co-ordinates to camera co-ordinates
        pcl_out_rot[idx] = world2cam(pcl_out, view_x[:,idx],
                view_y[:,idx], args.batch_size, args.OUTPUT_PCL_SIZE)
        # Perspective transform
        pcl_out_persp[idx] = perspective_transform(pcl_out_rot[idx],
                args.batch_size)
        # 3D to 2D Projection
        proj_pred[idx] = cont_proj(pcl_out_persp[idx], args.grid_h, args.grid_w,
                args.SIGMA_SQ)

        # Loss
        loss_bce[idx], fwd[idx], bwd[idx] = get_loss_proj(proj_pred[idx], 
                proj_gt[:,idx],'bce_prob', 1.0, True, grid_dist_tensor, args)
        loss_fwd[idx] = 1e-4*tf.reduce_mean(fwd[idx])
        loss_bwd[idx] = 1e-4*tf.reduce_mean(bwd[idx])

        if not args.ONLY_BCE:
            loss += args.lambda_bce*tf.reduce_mean(loss_bce[idx]) +\
                        args.lambda_aff_fwd*loss_fwd[idx] + args.lambda_aff_bwd*loss_bwd[idx]
        else:
            loss += args.lambda_bce*tf.reduce_mean(loss_bce[idx]) 

    # Regularization loss
    if args.lambda_reg > 0:
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_loss)
        loss = (loss / args.N_VIEWS) + (args.lambda_reg*reg_loss)
    else:
        loss = (loss / args.N_VIEWS) 

    train_vars = [var for var in tf.global_variables() if 'recon_net' in 
            var.name]

    # Optimizer 
    opt = tf.train.AdamOptimizer(args.lr, beta1=0.9)       
    optim = opt.minimize(loss, var_list=train_vars)

    # Training params
    start_epoch = 0

    # Define savers to load and store models
    saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=2)
    
    # Add Tensorboard summaries
    loss_summ = []
    for idx, name in enumerate(loss_names):
        loss_summ.append(tf.summary.scalar(name, train_loss_summ[idx]))
    train_summ = tf.summary.merge(loss_summ)
    
    # GPU configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Run session
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(logs_folder, sess.graph_def)
        print 'Session started'
        print 'running initializer'
        sess.run(tf.global_variables_initializer())
        print 'done'
        
        # Load previous checkpoint
        init_flag = True
        st_batches = 0
        ckpt = tf.train.get_checkpoint_state(snapshot_folder)
        if ckpt is not None:
            print ('loading '+ckpt.model_checkpoint_path + '  ....')
            saver.restore(sess, ckpt.model_checkpoint_path)
            pdb.set_trace()
            start_iters = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1))

            start_epoch = int(start_iters/batches)
            st_batches = start_iters % batches
            init_flag = False
        
        since = time.time()
        print '*'*30,'\n','Training Started !!!\n', '*'*30
        
        if start_epoch+st_batches == 0:
            with open(log_file, 'w') as f:
                f.write(' '.join(['Epoch','Train_loss','BCE','Aff_Fwd','Aff_Bwd', 'Minutes','Seconds','\n']))

        # Initialize values to 0
        train_loss_N, fwd_loss_N, L_BCE_N, L_fwd_N, L_bwd_N = [0.]*5
        batch_out_mean = [0.]*4

        for i in xrange(start_epoch, args.max_epoch+1):
            random.shuffle(train_pair_indices)
            train_epoch_loss, train_epoch_bce, train_epoch_fwd, train_epoch_bwd = [0.]*4
            epoch_out = [train_epoch_loss, train_epoch_bce, train_epoch_fwd,
                train_epoch_bwd]
        
            if init_flag:
                st_batches = 0

            for b in xrange(st_batches, batches):
                global_step = i*batches + b + 1
                if args.CORR:
                    batch_data = fetch_batch_drc_corresp(train_models, 
                            train_pair_indices, b, BATCH_SIZE, args)
                else:
                    batch_data = fetch_batch_drc(train_models, 
                            train_pair_indices, b, BATCH_SIZE, args)
                batch_ip, batch_gt, model_names, batch_views, batch_K, batch_x, batch_y = batch_data
                batch_pcl_gt = fetch_batch_pcl_drc(train_models_pcl,
                        train_pair_indices, b, BATCH_SIZE)

                # change GT PCL co-ordinate axis to align with that of renderer
                batch_pcl_gt = preprocess_pcl_gt(batch_pcl_gt)

                feed_dict = {img_inp: batch_ip, proj_gt: batch_gt, 
                        pcl_gt: batch_pcl_gt, view_x: batch_x, view_y: batch_y}

                L, L_bce, L_fwd, L_bwd, _ = sess.run([loss, loss_bce, loss_fwd,
                    loss_bwd, optim], feed_dict)

                L_bce = np.asarray([item for item in L_bce.values()]).mean()
                L_fwd = np.asarray([item for item in L_fwd.values()]).mean()
                L_bwd = np.asarray([item for item in L_bwd.values()]).mean()

                batch_out = [L, L_bce, L_fwd, L_bwd]
                # Use loss values averaged over N batches for logging
                batch_out_mean = average_stats(batch_out_mean, batch_out,
                        b%args.print_n)
                train_loss_N, L_bce_N, L_fwd_N, L_bwd_N = batch_out_mean
                epoch_out = [epoch_item+(batch_item/batches) for
                        (epoch_item, batch_item) in zip(epoch_out, batch_out)]

                if global_step % args.print_n == 0:
                    feed_dict_summ = {}
                    for idx, item in enumerate(batch_out_mean):
                        feed_dict_summ[train_loss_summ[idx]] = item
                    summ = sess.run(train_summ, feed_dict_summ)
                    # Add to tensorboard summary
                    train_writer.add_summary(summ, global_step)

                    time_elapsed = time.time() - since
                    print 'Iters:%d, Total: %.4f, BCE: %.4f, Aff_Fwd: %.4f, Aff_Bwd: %.4f, Time: %d'\
                        %(global_step, train_loss_N, L_bce_N, L_fwd_N, L_bwd_N,
                                time_elapsed//60)

                if global_step % args.save_n == 0:
                    _proj, _loss_bce,_loss_fwd, _loss_bwd, _pcl_out = sess.run(
                            [proj_pred, loss_bce, fwd, bwd, pcl_out], feed_dict)
                    save_outputs(proj_images_folder, global_step, _proj,
                        _loss_bce, _loss_fwd, _loss_bwd, batch_gt, model_names)
                    save_outputs_pcl(proj_pcl_folder, global_step, _pcl_out,
                        model_names)

                if global_step % args.save_model_n == 0:
                    print 'Saving Model ....................'
                    saver.save(sess, join(snapshot_folder,
                        'model'), global_step=global_step)
                    print '..................... Model Saved'

            time_elapsed = time.time() - since
            train_epoch_loss, train_epoch_bce, train_epoch_fwd, train_epoch_bwd = epoch_out
            epoch_str = 'TRAIN Loss: {:.6f}  BCE: {:.6f}  FWD: {:.6f}  BWD: {:.6f}  Time:{:.0f}m\ {:.0f}s'.format(\
                train_epoch_loss, train_epoch_bce, train_epoch_fwd, 
                train_epoch_bwd, time_elapsed//60, time_elapsed%60)
            with open(log_file, 'a') as f:
                f.write(epoch_str+'\n')

            print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
            print epoch_str
            print '-'*140, '\n'
