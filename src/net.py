import tensorflow as tf
import tflearn

def recon_net_large(img_inp, FLAGS):
    x=img_inp
    #128 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x1=x
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #64 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x2=x
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #32 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x3=x
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #16 16
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x4=x
    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,FLAGS.bottleneck,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,256,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,256,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,FLAGS.OUTPUT_PCL_SIZE*3,activation='linear',weight_decay=1e-3,regularizer='L2')
    x=tf.reshape(x,(-1,FLAGS.OUTPUT_PCL_SIZE,3))
    x = tf.nn.tanh(x)
    return x
