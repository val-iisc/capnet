import tensorflow as tf
import numpy as np
import pdb


def cont_proj(pcl, grid_h, grid_w, sigma_sq=0.5):
    '''
    Continuous approximation of Orthographic projection of point cloud
    to obtain Silhouette
    args:
            pcl: float, (N_batch,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            grid_h, grid_w: int, ();
                     output depth map height and width
    returns:
            grid_val: float, (N_batch,H,W); 
                      output silhouette
    '''
    x, y, z = tf.split(pcl, 3, axis=2)
    pcl_norm = tf.concat([x, y, z], 2)
    pcl_xy = tf.concat([x,y], 2)
    out_grid = tf.meshgrid(tf.range(grid_h), tf.range(grid_w), indexing='ij')
    out_grid = [tf.to_float(out_grid[0]), tf.to_float(out_grid[1])]
    grid_z = tf.expand_dims(tf.zeros_like(out_grid[0]), axis=2) # (H,W,1)
    grid_xyz = tf.concat([tf.stack(out_grid, axis=2), grid_z], axis=2)  # (H,W,3)
    grid_xy = tf.stack(out_grid, axis=2)                # (H,W,2)
    grid_diff = tf.expand_dims(tf.expand_dims(pcl_xy, axis=2), axis=2) - grid_xy # (BS,N_PTS,H,W,2) 
    grid_val = apply_kernel(grid_diff, sigma_sq)    # (BS,N_PTS,H,W,2) 
    grid_val = grid_val[:,:,:,:,0]*grid_val[:,:,:,:,1]  # (BS,N_PTS,H,W) 
    grid_val = tf.reduce_sum(grid_val, axis=1)          # (BS,H,W)
    grid_val = tf.nn.tanh(grid_val)
    return grid_val


def disc_proj(pcl, grid_h, grid_w):
    '''
    Discrete Orthographic projection of point cloud
    to obtain Silhouette 
    Handles only batch size 1 for now
    args:
            pcl: float, (N_batch,N_Pts,3); input point cloud
                     values assumed to be in (-1,1)
            grid_h, grid_w: int, ();
                     output depth map height and width
    returns:
            grid_val: float, (N_batch,H,W); output silhouette
    '''
    x, y, z = tf.split(pcl, 3, axis=2)
    pcl_norm = tf.concat([x, y, z], 2)
    pcl_xy = tf.concat([x,y], 2)
    xy_indices = tf.to_int32(pcl_xy)
    xy_values = tf.ones_like(xy_indices[:,:,0])
    xy_shape = (grid_h, grid_w)
    out_grid = tf.scatter_nd(xy_indices[0], xy_values[0], xy_shape)
    out_grid = tf.expand_dims(out_grid, axis=0)
    return out_grid


def apply_kernel(x, sigma_sq=0.5):
    '''
    Get the un-normalized gaussian kernel with point co-ordinates as mean and 
    variance sigma_sq
    args:
            x: float, (BS,N_PTS,H,W,2); mean subtracted grid input 
            sigma_sq: float, (); variance of gaussian kernel
    returns:
            out: float, (BS,N_PTS,H,W,2); gaussian kernel
    '''
    out = (tf.exp(-(x**2)/(2.*sigma_sq)))
    return out


def perspective_transform(xyz, batch_size):
    '''
    Perspective transform of pcl; Intrinsic camera parameters are assumed to be
    known (here, obtained using parameters of GT image renderer, i.e. Blender)
    Here, output grid size is assumed to be (64,64) in the K matrix
    TODO: use output grid size as argument
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
    returns:
            xyz_out: float, (BS,N_PTS,3); perspective transformed point cloud 
    '''
    K = np.array([
            [120., 0., -32.],
            [0., 120., -32.],
            [0., 0., 1.]]).astype(np.float32)
    K = np.expand_dims(K, 0)
    K = np.tile(K, [batch_size,1,1])

    xyz_out = tf.matmul(K, tf.transpose(xyz, [0,2,1]))
    xy_out = xyz_out[:,:2]/abs(tf.expand_dims(xyz[:,:,2],1))
    xyz_out = tf.concat([xy_out, abs(xyz_out[:,2:])],axis=1)
    return tf.transpose(xyz_out, [0,2,1])


def world2cam(xyz, az, el, batch_size, N_PTS=1024):
    '''
    Convert pcl from world co-ordinates to camera co-ordinates
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            az: float, (BS); azimuthal angle of camera in radians
            elevation: float, (BS); elevation of camera in radians
            batch_size: int, (); batch size
            N_PTS: float, (); number of points in point cloud
    returns:
            xyz_out: float, (BS,N_PTS,3); output point cloud in camera
                        co-ordinates
    '''
    # Distance of object from camera - fixed to 2
    d = 2.
    # Calculate translation params
    # Camera origin calculation - az,el,d to 3D co-ord
    tx, ty, tz = [0, 0, d]
    rotmat_az=[
                [tf.ones_like(az),tf.zeros_like(az),tf.zeros_like(az)],
                [tf.zeros_like(az),tf.cos(az),-tf.sin(az)],
                [tf.zeros_like(az),tf.sin(az),tf.cos(az)]
                ]

    rotmat_el=[
                [tf.cos(el),tf.zeros_like(az), tf.sin(el)],
                [tf.zeros_like(az),tf.ones_like(az),tf.zeros_like(az)],
                [-tf.sin(el),tf.zeros_like(az), tf.cos(el)]
                ]

    rotmat_az = tf.transpose(tf.stack(rotmat_az, 0), [2,0,1])
    rotmat_el = tf.transpose(tf.stack(rotmat_el, 0), [2,0,1])
    rotmat = tf.matmul(rotmat_el, rotmat_az)

    tr_mat = tf.tile(tf.expand_dims([tx, ty, tz],0), [batch_size,1]) # [B,3]
    tr_mat = tf.expand_dims(tr_mat,2) # [B,3,1]
    tr_mat = tf.transpose(tr_mat, [0,2,1]) # [B,1,3]
    tr_mat = tf.tile(tr_mat,[1,N_PTS,1]) # [B,1024,3]

    xyz_out = tf.matmul(rotmat,tf.transpose((xyz),[0,2,1])) - tf.transpose(tr_mat,[0,2,1])

    return tf.transpose(xyz_out,[0,2,1])
