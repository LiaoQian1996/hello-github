from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb

def clip(var):
    x = tf.ones_like(var)
    y = tf.zeros_like(var)
    var = tf.where(var > 1.0, x, var)
    var = tf.where(var < -1.0, y, var)
    return var

def create_local_patches(layer, patch_size, padding='VALID'):
    # type: (tf.Tensor, int, str) -> tf.Tensor
    """

    :param layer: Feature layer tensor with dimension (1, height, width, feature)
    :param patch_size: The width and height of the patch. It is set to 3 in the paper https://arxiv.org/abs/1601.04589
    :param padding: a string representing the padding style.
    :return: Patches with dimension (cardinality, patch_size, patch_size, feature)
    """
    return tf.extract_image_patches(layer, ksizes=[1, patch_size, patch_size, 1],
                                    strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding=padding)


def patch_matching(generated_layer_patches, style_layer_patches, patch_size):
    # type: (tf.Tensor, tf.Tensor, int) -> tf.Tensor
    """
    The patch matching is implemented as an additional convolutional layer for fast computation.
    In this case patches sampled from the style image are treated as the filters.
    :param generated_layer_patches: Size (batch, height, width, patch_size * patch_size * feature)
    :param style_layer_patches:Size (1, height, width, patch_size * patch_size * feature)
    :param patch_size: the patch size for mrf.
    :return: Best matching patch with size (batch, height, width, patch_size * patch_size * feature)
    """
    # Every patch and every feature layer are treated as equally important after normalization.
    normalized_generated_layer_patches = tf.nn.l2_normalize(generated_layer_patches, dim=[3])
    normalized_style_layer_patches = tf.nn.l2_normalize(style_layer_patches, dim=[3])
    # A better way to do this is to treat them as convolutions.
    # They have to be in dimension
    # (height * width, patch_size, patch_size, feature) <=> (batch, in_height, in_width, in_channels)
    # (patch_size, patch_size, feature, height * width) <= > (filter_height, filter_width, in_channels, out_channels)
    # Initially they are in [batch, out_rows, out_cols, patch_size * patch_size * depth]
    original_shape = normalized_style_layer_patches.get_shape().as_list()
    height = original_shape[1]
    width = original_shape[2]
    depth = int(original_shape[3] / patch_size / patch_size)
    normalized_style_layer_patches = tf.squeeze(normalized_style_layer_patches)

    normalized_style_layer_patches = tf.reshape(normalized_style_layer_patches,
                                                [height, width, patch_size, patch_size, depth])
    normalized_style_layer_patches = tf.reshape(normalized_style_layer_patches,
                                                [height * width, patch_size, patch_size, depth])
    normalized_style_layer_patches = tf.transpose(normalized_style_layer_patches, perm=[1, 2, 3, 0])
    style_layer_patches_reshaped = tf.reshape(style_layer_patches, [height, width, patch_size, patch_size, depth])
    style_layer_patches_reshaped = tf.reshape(style_layer_patches_reshaped,
                                              [height * width, patch_size, patch_size, depth])
    print("type of before : ",normalized_generated_layer_patches)
    normalized_generated_layer_patches_per_batch = tf.unstack(normalized_generated_layer_patches, axis=0)
    print("type of after : ",normalized_generated_layer_patches_per_batch)
    ret = []
    for batch in normalized_generated_layer_patches_per_batch:
        original_shape = batch.get_shape().as_list()
        height = original_shape[0]
        width = original_shape[1]
        depth = int(original_shape[2] / patch_size / patch_size)
        batch = tf.squeeze(batch)
        #height, width, patch_size, patch_size, depth = [height]
        batch = tf.reshape(batch, [height, width, patch_size, patch_size, depth])
        batch = tf.reshape(batch, [height * width, patch_size, patch_size, depth])
        # According to images-analogies github, for cross-correlation, we should flip the kernels
        # That is normalized_style_layer_patches should be [:, ::-1, ::-1, :]
        # I didn't see that in any other source, nor do I see why I should do so.
        convs = tf.nn.conv2d(batch, normalized_style_layer_patches, strides=[1, 1, 1, 1], padding='VALID')
        argmax = tf.squeeze(tf.argmax(convs, axis=3))
        # best_match has shape [height * width, patch_size, patch_size, depth]
        best_match = tf.gather(style_layer_patches_reshaped, indices=argmax)
        best_match = tf.reshape(best_match, [height, width, patch_size, patch_size, depth])
        best_match = tf.reshape(best_match, [height, width, patch_size * patch_size * depth])
        ret.append(best_match)
    ret = tf.stack(ret)
    return ret


def total_variation_loss(image):
    tv_y_size = tf.size(image[:,1:,:,:],out_type=tf.float32)
    tv_x_size = tf.size(image[:,:,1:,:],out_type=tf.float32)
    tv_loss =   (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:-1,:]) /
                    tv_x_size))
    return tv_loss

def get_layer_scope(type):
    if type == 'VGG11':
        target_layer = 'vgg_19/conv1/conv1_1'  
    elif type == 'VGG12':
        target_layer = 'vgg_19/conv1/conv1_2'  
    elif type == 'VGG21':
        target_layer = 'vgg_19/conv2/conv2_1'  
    elif type == 'VGG31':
        target_layer = 'vgg_19/conv3/conv3_1'
    elif type == 'VGG41':
        target_layer = 'vgg_19/conv4/conv4_1'
    elif type == 'VGG51':
        target_layer = 'vgg_19/conv5/conv5_1'
    elif type == 'VGG54':
        target_layer = 'vgg_19/conv5/conv5_4'        
    else:
        raise NotImplementedError('Unknown layer name')
    return target_layer

def get_style_layer_list(layer,single_layer=False):
    style_layers = []
    if single_layer == True:
        if layer == 'VGG11':
            style_layers = ['VGG11']
        elif layer == 'VGG21':
            style_layers = ['VGG21']
        elif layer == 'VGG31':
            style_layers = ['VGG31']
        elif layer == 'VGG41':
            style_layers = ['VGG41']
        elif layer == 'VGG51':
            style_layers = ['VGG51']
        elif layer == 'VGG54':
            style_layers = ['VGG54']
        else:
            raise ValueError("NO THIS LAYER !")
    else:
        if layer == 'VGG11':
            style_layers = ['VGG11']
        elif layer == 'VGG21':
            style_layers = ['VGG11','VGG21']
        elif layer == 'VGG31':
            style_layers = ['VGG11','VGG21','VGG31']
        elif layer == 'VGG41':
            style_layers = ['VGG11','VGG21','VGG31','VGG41']
        elif layer == 'VGG51':
            style_layers = ['VGG11','VGG21','VGG31','VGG41','VGG51']
        elif layer == 'VGG54':
            style_layers = ['VGG11','VGG21','VGG31','VGG41','VGG51','VGG54']
        else:
            raise ValueError(" No such layer in layer list.")
    return style_layers   

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2
    
# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    FLAGS = vars(FLAGS)
    for name, value in sorted(FLAGS.items()):
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))
    print('End of configuration')

# VGG19 component
def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

# VGG19 net
"""Oxford Net VGG 19-Layers version E Example.
Note: All the fully_connected layers have been transformed to conv2d layers.
    To use in classification mode, resize input to 224x224.
Args:
inputs: a tensor of size [batch_size, height, width, channels].
num_classes: number of predicted classes.
is_training: whether or not the model is being trained.
dropout_keep_prob: the probability that activations are kept in the dropout
  layers during training.
spatial_squeeze: whether or not should squeeze the spatial dimensions of the
  outputs. Useful to remove unnecessary dimensions for classification.
scope: Optional scope for the variables.
fc_conv_padding: the type of padding to use for the fully connected layer
  that is implemented as a convolutional layer. Use 'SAME' padding if you
  are applying the network in a fully convolutional manner and want to
  get a prediction map downsampled by a factor of 32 as an output. Otherwise,
  the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
Returns:
the last op containing the log predictions and end_points dict.
"""
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
    
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    return net,end_points
vgg_19.default_image_size = 224
