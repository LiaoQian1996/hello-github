from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
import collections
import os
import math
import scipy.misc as sic
from PIL import Image
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib as contrib
    
def generator(FLAGS,target,initial=None,reuse=False,training=True):
    if initial is None:
        if FLAGS.texture_shape == [-1,-1]:
            shape = [1,target.shape[1],target.shape[2],3]
            var = tf.get_variable('gen_img',shape=shape, \
                 initializer = tf.truncated_normal_initializer(0,0.2),\
                               dtype=tf.float32,trainable=True, collections=None)            
        else:
            shape = [1,FLAGS.texture_shape[0],FLAGS.texture_shape[1],3]
            var = tf.get_variable('gen_img',shape=shape, \
                 initializer = tf.truncated_normal_initializer(0,0.2),\
                               dtype=tf.float32,trainable=True, collections=None)
    else:
        var = tf.get_variable('gen_img', \
                    initializer = initial)
    return var
    
# Define the dataloader
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        image_raw = Image.open(FLAGS.target_dir)
        if image_raw.mode is not 'RGB':
            image_raw = image_raw.convert('RGB')
        image_raw = np.asarray(image_raw)/255
        targets = tf.constant(image_raw)  
        targets = tf.image.convert_image_dtype(targets, dtype = tf.float32, saturate = True)
        targets = preprocess(targets)  
        samples = tf.expand_dims(targets, axis=0) 
        contents_raw = np.asarray(Image.open(FLAGS.content_dir))/255
        contents = tf.constant(contents_raw)
        contents = tf.image.convert_image_dtype(contents, dtype=tf.float32,saturate=True)
        contents = preprocess(contents)  
        contents = tf.expand_dims(contents,axis=0)            
    return samples, contents

def Optimizer(targets,initials,contents,FLAGS=None):
    # Define the container of the parameter
    Procedure = collections.namedtuple('Procedure', 'optimizer, content_loss, style_loss,tv_loss,\
                                        outputs, global_step, \
                                        learning_rate')

    # Build the generator part
    with tf.variable_scope('generator'):
        gen_output = generator(FLAGS,targets,initials,reuse=False)

    # Calculating the generator loss
    with tf.name_scope('generator_loss'):
        # Content loss
        '''
        with tf.name_scope('content_loss'):
            # Compute the euclidean distance between the two features
            # check=tf.equal(extracted_feature_gen, extracted_feature_target)
            diff = targets - gen_output
            content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
        '''
        
        with tf.name_scope('tv_loss'):
            tv_loss = total_variation_loss(gen_output)
        
        with tf.name_scope('mean_loss'):
            mean_diff = tf.reduce_mean(gen_output)-tf.reduce_mean(targets)
            mean_loss = tf.abs(mean_diff)
        
        with tf.name_scope('mrf_loss'):
            def compute_mrf_loss(style_layer, generated_layer, patch_size=3, name=''):
            # type: (tf.Tensor, tf.Tensor, int, str) -> tf.Tensor
                """
                :param style_layer: The vgg feature layer by feeding it the style image.
                :param generated_layer: The vgg feature layer by feeding it the generated image.
                :param patch_size: The patch size of the mrf.
                :param name: Name scope of this loss.
                :return: the mrf loss between the two inputted layers represented as a scalar tensor.
                """
                generated_layer_patches = create_local_patches(generated_layer, patch_size)
                style_layer_patches = create_local_patches(style_layer, patch_size)
                generated_layer_nn_matched_patches = patch_matching(generated_layer_patches, style_layer_patches, patch_size)
                _, height, width, number = map(lambda i: i.value, generated_layer.get_shape())
                size = height * width * number
                # Normalize by the size of the image as well as the patch area.
                mrf_loss = tf.div(tf.reduce_sum(tf.square(generated_layer_patches - generated_layer_nn_matched_patches)),
                              size * (patch_size ** 2))
                return mrf_loss 
            
            _, vgg_gen_output = vgg_19(gen_output,is_training=False,reuse=False)
            _, vgg_tar_output = vgg_19(targets,is_training=False,reuse=True)
            '''首先，将一个batch的图像输入vgg_19中，得到一系列不同层的输出'''
            mrf_loss = tf.zeros([])
            tar_layer = FLAGS.top_style_layer
            target_layer = get_layer_scope(tar_layer)
            gen_feature = vgg_gen_output[target_layer]
            tar_feature = vgg_tar_output[target_layer]            
            mrf_loss = compute_mrf_loss(tar_feature, gen_feature, patch_size=3, name='')

        '''
        with tf.name_scope('style_loss'):
            def gram(features):
                features = tf.reshape(features,[-1,features.shape[3]])
                return tf.matmul(features,features,transpose_a=True)\
                             / tf.cast(features.shape[0]*features.shape[1],dtype=tf.float32)
            
            def new_style_loss(gen_img,targets,style_layer_list,reuse=True):
                _, vgg_gen_output = vgg_19(gen_img,is_training=False,reuse=True)
                _, vgg_tar_output = vgg_19(targets,is_training=False,reuse=True)
                sl = tf.zeros([])
                ratio_list=[100.0,10.0,1.0,0.01,100.0,2500.0]
                for i in range(len(style_layer_list)):
                    tar_layer=style_layer_list[i]
                    target_layer = get_layer_scope(tar_layer)
                    gen_feature = vgg_gen_output[target_layer]
                    tar_feature = vgg_tar_output[target_layer]
                    sl = sl + tf.nn.l2_loss(gram(gen_feature)-gram(tar_feature)) *ratio_list[i] 
                return sl
            
            #gram_loss = new_style_loss(gen_output,targets,\
                                        #get_style_layer_list('VGG31'))
        '''
        style_loss = mrf_loss
        #gen_loss = style_loss + gram_loss + FLAGS.W_tv * tv_loss 
        gen_loss = style_loss + FLAGS.W_tv * tv_loss 
        
    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.train.get_or_create_global_step() 
        '''
        boundaries = [1000,20000,30000]
        learning_rates = [0.1,0.01,0.001,0.0001]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries,\
                                                   values=learning_rates)
        '''
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,\
                                                   FLAGS.decay_step, FLAGS.decay_rate,\
                                                   staircase = FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)   
        
    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).\
            minimize(gen_loss, global_step = global_step, var_list = gen_tvars)
    
    return Procedure(
        tv_loss = tv_loss,
        style_loss = style_loss,
        content_loss = tf.zeros([]),
        outputs = gen_output,
        optimizer = tf.group(incr_global_step, optimizer),
        global_step = global_step,
        learning_rate = learning_rate
    )