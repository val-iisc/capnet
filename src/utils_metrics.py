###############################################################################
# Functions needed in calculating metrics
###############################################################################

import os, sys
from os.path import join
import numpy as np
import tensorflow as tf
import tf_nndistance
from tf_auctionmatch import auction_match
import json
from itertools import product
from shapenet_taxonomy import shapenet_category_to_id
import cv2
import pdb

PNG_FILES = ['render_0.png', 'render_1.png', 'render_2.png', 'render_3.png',
        'render_4.png', 'render_5.png', 'render_6.png', 'render_7.png',
        'render_8.png', 'render_9.png']

# xyz is a batch
def rotate(xyz, xangle=0, yangle=0):
    '''
    Rotate input pcl along x and y axes using tensorflow
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
            xangle, yangle: float, (); angles by which pcl has to be rotated, 
                                    in radians
    returns:
            xyz: float, (BS,N_PTS,3); rotated point clooud
    '''
    xangle = np.pi*xangle/180
    yangle = np.pi*yangle/180
    rotmat = np.eye(3)
    BATCH_SIZE = (xyz.shape)[0]

    rotmat=rotmat.dot(np.array([
	    [1.0,0.0,0.0],
	    [0.0,np.cos(xangle),-np.sin(xangle)],
	    [0.0,np.sin(xangle),np.cos(xangle)],
	    ]))

    rotmat=rotmat.dot(np.array([
	    [np.cos(yangle),0.0,-np.sin(yangle)],
	    [0.0,1.0,0.0],
	    [np.sin(yangle),0.0,np.cos(yangle)],
	    ]))

    _rotmat = tf.constant(rotmat, dtype=tf.float32)
    _rotmat = tf.reshape(tf.tile(_rotmat,(BATCH_SIZE,1)), shape=(BATCH_SIZE,3,3))

    return tf.matmul(xyz,_rotmat)


# xyz is a single pcl
def np_rotate(xyz, xangle=0, yangle=0, inverse=False):
    '''
    Rotate input pcl along x and y axes using numpy
    args:
            xyz: float, (N_PTS,3), numpy array; input point cloud
            xangle, yangle: float, (); angles by which pcl has to be rotated, 
                                    in radians
    returns:
            xyz: float, (N_PTS,3); rotated point clooud
    '''
    rotmat = np.eye(3)
    rotmat=rotmat.dot(np.array([
	    [1.0,0.0,0.0],
	    [0.0,np.cos(xangle),-np.sin(xangle)],
	    [0.0,np.sin(xangle),np.cos(xangle)],
	    ]))
    rotmat=rotmat.dot(np.array([
	    [np.cos(yangle),0.0,-np.sin(yangle)],
	    [0.0,1.0,0.0],
	    [np.sin(yangle),0.0,np.cos(yangle)],
	    ]))
    if inverse:
	    rotmat = np.linalg.inv(rotmat)
    return xyz.dot(rotmat)


def scale(gt_pc, pr_pc): 
    '''
    Scale GT and predicted PCL to a bounding cube with edges from [-0.5,0.5] in
    each axis. 
    args:
            gt_pc: float, (BS,N_PTS,3); GT point cloud
            pr_pc: float, (BS,N_PTS,3); predicted point cloud
    returns:
            gt_scaled: float, (BS,N_PTS,3); scaled GT point cloud
            pred_scaled: float, (BS,N_PTS,3); scaled predicted point cloud
    '''
    pred = tf.cast(pr_pc, dtype=tf.float32)
    gt   = tf.cast(gt_pc, dtype=tf.float32)

    min_gt = tf.convert_to_tensor([tf.reduce_min(gt[:,:,i], axis=1) for i in xrange(3)])
    max_gt = tf.convert_to_tensor([tf.reduce_max(gt[:,:,i], axis=1) for i in xrange(3)])
    min_pr = tf.convert_to_tensor([tf.reduce_min(pred[:,:,i], axis=1) for i in xrange(3)])
    max_pr = tf.convert_to_tensor([tf.reduce_max(pred[:,:,i], axis=1) for i in xrange(3)])

    length_gt = tf.abs(max_gt - min_gt)
    length_pr = tf.abs(max_pr - min_pr)

    diff_gt = tf.reduce_max(length_gt, axis=0, keep_dims=True) - length_gt
    diff_pr = tf.reduce_max(length_pr, axis=0, keep_dims=True) - length_pr

    new_min_gt = tf.convert_to_tensor([min_gt[i,:] - diff_gt[i,:]/2. for i in xrange(3)])
    new_max_gt = tf.convert_to_tensor([max_gt[i,:] + diff_gt[i,:]/2. for i in xrange(3)])
    new_min_pr = tf.convert_to_tensor([min_pr[i,:] - diff_pr[i,:]/2. for i in xrange(3)])
    new_max_pr = tf.convert_to_tensor([max_pr[i,:] + diff_pr[i,:]/2. for i in xrange(3)])

    size_pr = tf.reduce_max(length_pr, axis=0)
    size_gt = tf.reduce_max(length_gt, axis=0)

    scaling_factor_gt = 1. / size_gt # 2. is the length of the [-1,1] cube
    scaling_factor_pr = 1. / size_pr

    box_min = tf.ones_like(new_min_gt) * -0.5

    adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt
    adjustment_factor_pr = box_min - scaling_factor_pr * new_min_pr

    pred_scaled = tf.transpose((tf.transpose(pred) * scaling_factor_pr)) + tf.reshape(tf.transpose(adjustment_factor_pr), (-1,1,3))
    gt_scaled   = tf.transpose((tf.transpose(gt) * scaling_factor_gt)) + tf.reshape(tf.transpose(adjustment_factor_gt), (-1,1,3))
    return gt_scaled, pred_scaled


# new scale functions
def scale_pcl(points, max_length=1.):
    BATCH_SIZE = (points.shape)[0]
    valid_array = tf.ones_like(points)*0.5
    valid = tf.greater(valid_array, tf.abs(points))
    out_pcls = []
    pc_valids = []
    for b in xrange(BATCH_SIZE):
        pc = points[b]
        val = valid[b]
        val = tf.logical_and(tf.logical_and(val[:,0],val[:,1]),val[:,2])
        pc_valid = tf.boolean_mask(pc,val)
        pc_valids.append(pc_valid)
        bound_l = tf.reduce_min(pc_valid, axis=0, keep_dims=True)
        bound_h = tf.reduce_max(pc_valid, axis=0, keep_dims=True)
        pc = pc - (bound_l + bound_h) / 2.
        pc = pc / tf.reduce_max((bound_h - bound_l), keep_dims=True)
        pc = pc*max_length
        out_pcls.append(pc)
    out_pcls = tf.stack(out_pcls)
    return out_pcls


def scale_outlier(gt_pcl, pred_pcl):
    '''
    Scale GT and predicted PCL to a bounding cube with edges from [-0.5,0.5] in
    each axis. Ignores outliers while calculating scaling factor to avoid errors
    args:
            gt_pc: float, (BS,N_PTS,3); GT point cloud
            pr_pc: float, (BS,N_PTS,3); predicted point cloud
    returns:
            gt_scaled: float, (BS,N_PTS,3); scaled GT point cloud
            pred_scaled: float, (BS,N_PTS,3); scaled predicted point cloud
    '''
    gt_pcl_scaled = scale_pcl(gt_pcl)
    pred_pcl_scaled = scale_pcl(pred_pcl)
    return gt_pcl_scaled, pred_pcl_scaled


def get_metrics(gt_pcl, pred_pcl, args):
    '''
    Obtain chamfer and emd distances between GT and predicted pcl
    args:
            gt_pcl: float, (BS,N_PTS,3); GT point cloud
            pred_pcl: float, (BS,N_PTS,3); predicted point cloud
    returns:
            dists_forward: float, (BS); forward chamfer distance
            dists_backward: float, (BS); backward chamfer distance
            chamfer_distance: float, (BS); chamfer distance
            emd: float, (BS); earth mover's distance
    '''
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt_pcl, pred_pcl)
    dists_forward = tf.reduce_mean(dists_forward, axis=1) # (BATCH_SIZE,args.N_PTS) --> (BATCH_SIZE)
    dists_backward = tf.reduce_mean(dists_backward, axis=1)
    chamfer_distance = dists_backward + dists_forward

    X,_ = tf.meshgrid(range(args.batch_size), range(args.OUTPUT_PCL_SIZE), indexing='ij')
    ind, _ = auction_match(pred_pcl, gt_pcl) # Ind corresponds to points in pcl_gt
    ind = tf.stack((X, ind), -1)
    emd = tf.reduce_mean(tf.reduce_sum((tf.gather_nd(gt_pcl, ind) - pred_pcl)**2, axis=-1), axis=1) # (BATCH_SIZE,args.N_PTS,3) --> (BATCH_SIZE,args.N_PTS) --> (BATCH_SIZE)
    return dists_forward, dists_backward, chamfer_distance, emd


def get_averages(csv_path):
    '''
    Obtain average of values in a csv file
    args:
            csv_path: str; path for the csv file to be averaged
    '''
    column_sums = None
    with open(csv_path) as f:
	lines = f.readlines()[1:]
	rows_of_numbers = [map(float, line.split(';')[1:]) for line in lines]
	sums = map(sum, zip(*rows_of_numbers))
	averages = [sum_item / len(lines) for sum_item in sums]
	return averages


def load_previous_checkpoint(snapshot_folder, saver, sess):
    '''
    Load model from checkpoint
    args:
            snapshot_folder: str; checkpoint directory
            saver: Saver; saver object
            sess: Session; tensorflow session instance
    '''    
    ckpt = tf.train.get_checkpoint_state(snapshot_folder)
    if ckpt is not None:
	ckpt_path = ckpt.model_checkpoint_path
	model = ckpt_path.strip().split('/')[-1]
	ckpt_path = os.path.abspath(join(snapshot_folder, model))
	print ('loading '+ckpt_path + '  ....')
	saver.restore(sess, ckpt_path)


def save_screenshots(_gt_scaled, _pr_scaled, img, screenshot_dir, fid, eval_set,
        args):
    '''
    Save 2D projections of PCL from multiple predefined viewpoints
    args:
            _gt_scaled: float, (N_PTS,3); scaled GT point cloud
            _pr_scaled: float, (N_PTS,3); scaled predicted point cloud
            img: uint8, (H,W,3); input RGB image
            screenshot_dir: str; directory path for saving outputs
            fid: str; name of model being saved
            eval_set: str; data to be used, one of [train, val, test]
            args: arguments
    '''
    # clock, front, anticlock, side, back, top
    xangles = np.array([-50, 0, 50, 90, 180, 0]) * np.pi / 180.
    yangles = np.array([20, 20, 20, 20, 20, 90]) * np.pi / 180.

    gts = []
    results = []
    overlaps = []

    for xangle, yangle in zip(xangles, yangles):
        gt_rot = show3d_balls.get2D(np_rotate(_gt_scaled, xangle=xangle, 
            yangle=yangle), ballradius=args.ballradius)
        result_rot = show3d_balls.get2D(np_rotate(_pr_scaled, xangle=xangle, 
            yangle=yangle), ballradius=args.ballradius)
        overlap_rot = show3d_balls.get2Dtwopoints(np_rotate(_gt_scaled, 
            xangle=xangle, yangle=yangle), np_rotate(_pr_scaled, xangle=xangle,
                yangle=yangle), ballradius=args.ballradius)
        gts.append(gt_rot)
        results.append(result_rot)
        overlaps.append(overlap_rot)

    cv2.imwrite(join(screenshot_dir, '%s_%s_inp.png'%(eval_set, fid)), img)

    if args.save_screenshots:
        gt = np.concatenate(gts, 1)
        result = np.concatenate(results, 1)
        overlap = np.concatenate(overlaps, 1)
        final = np.concatenate((gt, result, overlap), 0)
        cv2.imwrite(join(screenshot_dir, '%s_%s.png'%(eval_set,fid)), final)

    if args.save_gifs:
        import imageio
        final = [np.concatenate((gt, result, overlap), 1) for gt, result, overlap in zip(gts, results, overlaps)]
        imageio.mimsave(join(screenshot_dir, '%s_%s.gif'%(eval_set,fid)), final, 'GIF', duration=0.7)
    return


def fetch_batch(models, indices, batch_num, batch_size, pcl_data_dir, args):
    ''' 
    Obtain batch of data samples - GT point clouds and input images
    args:
            models: list of all ids/names of pcl models
            indices: indices to be chosen from models for the current batch
            batch_num: index of the current batch
            batch_size: number of samples in a batch
            pcl_data_dir: root directory of ShapeNet_v1 point clouds
            args: input arguments from argparse
    returns:
            batch_ip: uint8, (BS,IMG_H,IMG_W,3); input rgb images
            batch_gt: float, (BS,N_PTS,3); GT point cloud
    '''

    batch_ip = []
    batch_gt = []
    for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
        model_path = models[ind[0]]
        img_path = join(model_path, PNG_FILES[ind[1]])
        pcl_filename = 'pcl_1024_fps_trimesh.npy'
        model_pcl_path = '/'.join(model_path.split('/')[-2:])
        pcl_path = join(pcl_data_dir, model_pcl_path, pcl_filename)
        pcl_gt = np.load(pcl_path)
        try:
            if args.natural:
                img_path = join(data_dir_224, category_id, 
                        model_path.split('/')[-1], PNG_FILES[ind[1]])
                ip_image = blendBg(img_path, bgImgsList, args.IMG_H, args.IMG_W)
            else:
                img_path = join(model_path, PNG_FILES[ind[1]])
                ip_image = cv2.imread(img_path)
                ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
        except:
            print img_path
            continue
        try:
            batch_ip.append(ip_image)
            batch_gt.append(pcl_gt)
        except:
            pass
    batch_ip = np.array(batch_ip)
    batch_gt = np.array(batch_gt)
    return batch_ip, batch_gt


# Returns model_ids
def fetch_batch_paths(models, indices, batch_num, batch_size):
    ''' 
    Obtain id of point cloud models; point cloud file names are same as id
    args:
            models: list of all ids/names of pcl models
            indices: indices to be chosen from models for the current batch
            batch_num: index of the current batch
            batch_size: number of samples in a batch
    returns:
            paths: str; model id of point cloud 
    '''
    paths = []
    for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
        model_path = models[ind[0]]
        try:
            l = model_path.strip().split('/')
            category_id = l[-2]
            model_id = l[-1]
            fid = '_'.join([l[-2], l[-1], str(ind[1])])
            paths.append(fid)
        except:
            print fid
            pass
    return paths


def get_drc_models(data_dir, args):
    '''
    Obtain model names and corresponding indices for all models in the dataset
    returns:
            models: list of str; names of models
            indices: list of tuple; indices of model and view-point
    '''
    models = []

    if args.category == 'all':
	    cats = ['chair','car','aero']
    else:
	    cats = [args.category]

    with open('src/corrupt_chairs.json', 'r') as f:
        corrupt_chairs = json.load(f)

    for cat in cats:
	category_id = shapenet_category_to_id[cat]
	with open(join(data_dir, 'splits', category_id+'_%s_list.txt'%args.eval_set)) as f:
	    for model in f.readlines():
		if args.natural:
            	    if model.strip() not in corrupt_chairs:
			models.append(join(data_dir,category_id,model.strip()))
		else:
		    models.append(join(data_dir,category_id,model.strip()))

    indices = list(product(xrange(len(models)), xrange(args.N_VIEWS)))
    print 'models={}  samples={}'.format(len(models),len(models)*args.N_VIEWS)
    print
    return models, indices

