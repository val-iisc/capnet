import os
import numpy as np
from itertools import product
from os.path import join, exists
import cv2
import scipy.io as sio
import scipy.misc as sc
import pdb


def create_folder(folders_list):
    '''
    Create empty directory if it doesn't already exist
    args:
            folders_list: list of str; list of directory paths
    '''
    for folder in folders_list:
	if not exists(folder):
	    os.makedirs(folder)


def get_shapenet_drc_models(data_dir, categs=['03001627'], num_views=10):
    '''
    Obtain indices and names of all point cloud models, for train and validation
    sets
    args:
            data_dir: str; root directory containing data of all categories
            categs: list of str; list of category ids for which indices and 
                            names have to be returned
            num_views: number of view points from which images are rendered for
                        each model
    returns:
            train_pair_indices, val_pair_indices: list of tuples of model index
                        and rendered image index
            train_models, val_models: list of str; names of training and
                        validation pcl models respectively
    '''
    train_models = []
    val_models = []
        
    for cat in categs:
        cat_train_model =\
        np.load(data_dir+'/splits/%s_train_list.npy'%cat)
        cat_val_model =\
        np.load(data_dir+'/splits/%s_val_list.npy'%cat)
        cat_train_model = [join(data_dir,cat,item) for item in cat_train_model]
        cat_val_model = [join(data_dir,cat,item) for item in cat_val_model]
        train_models.extend(cat_train_model)
        val_models.extend(cat_val_model)

    train_pair_indices = list(product(xrange(len(train_models)), xrange(num_views)))
    val_pair_indices = list(product(xrange(len(val_models)), xrange(num_views)))

    print 'TRAINING: models={}  samples={}'.format(len(train_models),len(train_models)*num_views)
    print 'VALIDATION: models={}  samples={}'.format(len(val_models),len(val_models)*num_views)
    print

    return train_models, val_models, train_pair_indices, val_pair_indices


def fetch_batch_drc(models, indices, batch_num, batch_size, FLAGS=None):
    '''
    Obtain a batch of data for training
    args:
            models: list of all ids/names of pcl models
            indices: indices to be chosen from models for the current batch
            batch_num: index of the current batch
            batch_size: number of samples in a batch
            FLAGS: input arguments while running the train.py file
    returns:
            All outputs are lists of length N_VIEWS. 
            Properties of each element in the list:
            batch_ip: uint8, (BS,IMG_H,IMG_W,3); input rgb images
            batch_gt: float, (BS,IMG_H,IMG_W); GT foreground masks
            batch_names: str; names of pcl models corresponding to input images
            batch_views: float, (4,4); camera extrinsic matrix
            batch_K: float, (3,3); camera intrinsic matrix
            batch_x: float, (); rotation angle along x-axis for the view point 
                                in radians
            batch_y: float, (); rotation angle along y-axis for the view point
                                in radians
    '''
    batch_ip = []
    batch_gt = []
    batch_names = []
    batch_views = []
    batch_K = []
    batch_x = []
    batch_y = []

    for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:

        model_gt = []
        model_view = []
        model_K = []
        model_x = []
        model_y = []
        model_path = models[ind[0]]
        model_name = model_path.split('/')[-1]
        img_path = join(model_path, 'render_%d.png'%ind[1])
        ip_image = cv2.imread(img_path)
        ip_image = cv2.resize(ip_image, (FLAGS.IMG_W,FLAGS.IMG_H))
        ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
        batch_ip.append(ip_image)
        batch_names.append(model_name)

        # Select the first N_VIEWS images
        for i in range(0,FLAGS.N_VIEWS):
            proj_path = join(model_path, 'depth_%d.png'%(i%10))
            view_path = join(model_path, 'camera_%d.mat'%(i%10))
            angles_path = join(model_path, 'view.txt')
            ip_proj = cv2.imread(proj_path)[:,:,0]
            ip_proj = cv2.resize(ip_proj, (FLAGS.IMG_W,FLAGS.IMG_H))
            ip_proj[ip_proj<254] = 1
            ip_proj[ip_proj>=254] = 0
            ip_proj = ip_proj.astype(np.float32)
            model_gt.append(ip_proj)
            view_proj = sio.loadmat(view_path)
            model_view.append(view_proj['extrinsic'])
            model_K.append(view_proj['K'])
            with open(angles_path, 'r') as fp:
                angles = [item.split('\n')[0] for item in fp.readlines()]
            angle = angles[i%10]
            angle_x = float(angle.split(' ')[0])
            angle_y = float(angle.split(' ')[1])
            model_x.append(angle_x*np.pi/180.)
            model_y.append(angle_y*np.pi/180.)
        batch_gt.append(model_gt)
        batch_views.append(model_view)
        batch_K.append(model_K)
        batch_x.append(model_x)
        batch_y.append(model_y)

    batch_ip = np.array(batch_ip)
    batch_gt = np.array(batch_gt)
    batch_views = np.array(batch_views)
    batch_K = np.array(batch_K)
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    return batch_ip, batch_gt, batch_names, batch_views, batch_K, batch_x, batch_y


def fetch_batch_drc_corresp(models, indices, batch_num, batch_size, FLAGS=None):
    '''
    Obtain a batch of data for training
    Fetch single mask from view-point corresponding to input image
    args:
            models: list of all ids/names of pcl models
            indices: indices to be chosen from models for the current batch
            batch_num: index of the current batch
            batch_size: number of samples in a batch
            FLAGS: input arguments while running the train.py file
    returns:
            All outputs are lists of length N_VIEWS. 
            Properties of each element in the list:
            batch_ip: uint8, (BS,IMG_H,IMG_W,3); input rgb images
            batch_gt: float, (BS,IMG_H,IMG_W); GT foreground masks
            batch_names: str; names of pcl models corresponding to input images
            batch_views: float, (4,4); camera extrinsic matrix
            batch_K: float, (3,3); camera intrinsic matrix
            batch_x: float, (); rotation angle along x-axis for the view point 
                                in radians
            batch_y: float, (); rotation angle along y-axis for the view point
                                in radians
    '''
    batch_ip = []
    batch_gt = []
    batch_names = []
    batch_views = []
    batch_K = []
    batch_x = []
    batch_y = []

    for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:

        model_gt = []
        model_view = []
        model_K = []
        model_x = []
        model_y = []
        model_path = models[ind[0]]
        model_name = model_path.split('/')[-1]
        img_path = join(model_path, 'render_%d.png'%ind[1])

        ip_image = cv2.imread(img_path)
        ip_image = cv2.resize(ip_image, (FLAGS.IMG_W,FLAGS.IMG_H))
        ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
        batch_ip.append(ip_image)
        batch_names.append(model_name)

        for i in range(ind[1],ind[1]+1):
            proj_path = join(model_path, 'depth_%d.png'%(i%10))
            view_path = join(model_path, 'camera_%d.mat'%(i%10))
            angles_path = join(model_path, 'view.txt')
            ip_proj = cv2.imread(proj_path)[:,:,0]
            ip_proj = cv2.resize(ip_proj, (FLAGS.IMG_W,FLAGS.IMG_H))
            ip_proj[ip_proj<254] = 1
            ip_proj[ip_proj>=254] = 0
            ip_proj = ip_proj.astype(np.float32)
            model_gt.append(ip_proj)
            view_proj = sio.loadmat(view_path)
            model_view.append(view_proj['extrinsic'])
            model_K.append(view_proj['K'])
            with open(angles_path, 'r') as fp:
                angles = [item.split('\n')[0] for item in fp.readlines()]
            angle = angles[i]
            angle_x = float(angle.split(' ')[0])
            angle_y = float(angle.split(' ')[1])
            model_x.append(angle_x*np.pi/180.)
            model_y.append(angle_y*np.pi/180.)
        batch_gt.append(model_gt)
        batch_views.append(model_view)
        batch_K.append(model_K)
        batch_x.append(model_x)
        batch_y.append(model_y)

    batch_ip = np.array(batch_ip)
    batch_gt = np.array(batch_gt)
    batch_views = np.array(batch_views)
    batch_K = np.array(batch_K)
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    return batch_ip, batch_gt, batch_names, batch_views, batch_K, batch_x, batch_y


def fetch_batch_pcl_drc(models, indices, batch_num, batch_size):
    '''
    Obtain batch of data samples - GT point clouds
    args:
            models: list of all ids/names of pcl models
            indices: indices to be chosen from models for the current batch
            batch_num: index of the current batch
            batch_size: number of samples in a batch
    returns:
            batch_gt: float, (BS,N_PTS,3); GT point cloud
    '''
    batch_gt = []
    for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:

        model_path = models[ind[0]]
        pcl_path_1K = join(model_path, 'pcl_1024_fps_trimesh.npy')
        pcl_gt = np.load(pcl_path_1K) ### change to 2K/16K as needed
        try:
            batch_gt.append(pcl_gt)
        except:
            pass
    batch_gt = np.array(batch_gt)
    return batch_gt


def save_outputs(out_dir, iters, _proj, _loss_bce, _loss_fwd, _loss_bwd,
        batch_gt, model_names):
    '''
    Save outputs during training
    args:
            out_dir: str; path for output directory
            iters: int, (); iteration number
            _proj: dict; dictionary of mask projections from each view
            _loss_bce, _loss_fwd, _loss_bwd: dict; dictionary of pixel-wise loss
                                values from each view
            batch_gt: list of uint8, (BS,N_VIEWS,IMG_H,IMG_W); GT masks
            model_names: list of str; names of models in current batch
            sess: Session object
    '''
    for k in range(len(_proj)):
        for l in range(len(_proj[0])):
            sc.imsave('%s/%s_%s_%s_pred.png'%(out_dir,iters,model_names[l],k),_proj[k][l])
            sc.imsave('%s/%s_%s_%s_gt.png'%(out_dir,iters,model_names[l],k),batch_gt[l][k])
            sc.imsave('%s/%s_%s_%s_bce_loss.png'%(out_dir,iters,model_names[l],k),_loss_bce[k][l])
            sc.imsave('%s/%s_%s_%s_aff_fwd.png'%(out_dir,iters,model_names[l],k),_loss_fwd[k][l])
            sc.imsave('%s/%s_%s_%s_aff_bwd.png'%(out_dir,iters,model_names[l],k),_loss_bwd[k][l])
    return True


def save_outputs_pcl(out_dir, iters, _pcl_out, model_names):
    '''
    Save point cloud outputs during training
    args:
            out_dir: str; path for output directory
            iters: int, (); iteration number
            pcl_out: float, (BS,N_PTS,3), numpy array; predicted point cloud 
            model_names: list of str; names of models in current batch
    '''
    for k in range(len(_pcl_out)):
        np.savetxt('%s/%s_%s_%s_pred.xyz'%(out_dir,model_names[k],iters,k),_pcl_out[k])
        np.save('%s/%s_%s_%s_pred.npy'%(out_dir,model_names[k],iters,k),_pcl_out[k])
    return True


def preprocess_pcl_gt(pcl):
    '''
    To align the GT pcl according to the axes of the GT image renderer(i.e.
    the co-ordinate system used while rendering the images from GT PCL),
    interchange the axes and change axes directions
    args:
            pcl: float, (BS,N_PTS,3), numpy array; input point cloud
    '''
    pcl[:,:,[0,2]] = pcl[:,:,[2,0]]
    pcl[:,:,[0,1]] = pcl[:,:,[1,0]]
    pcl[:,:,1] = -pcl[:,:,1]
    pcl[:,:,0] = -pcl[:,:,0]
    return pcl


def average_stats(val_mean, val_batch, iters):
    '''
    Update cumulative loss values
    args:
            val_mean: list of float; cumulative mean value
            val_batch: list of float; current value
            iters: iteration number
    returns:
            val_upd: list of float; updated cumulative mean values
    '''
    val_upd = [((item*iters)+batch_item)/(iters+1) for (item, batch_item) in\
            zip(val_mean, val_batch)]
    return val_upd
