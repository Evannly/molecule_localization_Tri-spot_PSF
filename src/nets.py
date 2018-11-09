'''
    Created on July, 2018
    
    author: chunyuan
'''

import tensorflow as tf


def LeakyReLU():
    def lrelu(x, alpha = 0.3):
        return tf.nn.relu(x)-alpha * tf.nn.relu(-x)
    return lrelu


def add_common_layers(y):
    y = tf.layers.batch_normalization(inputs=y)
    y = LeakyReLU()(y)
    return y

def deepLoco_decoder(x, img_channels=1, psize=64.0):
    '''

    :param inputs: input tensor, shape [?,nx,ny,img_channels]

    '''

    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    inputs = tf.reshape(x, tf.stack([-1,nx,ny,img_channels]))
    batch_size = tf.shape(inputs)[0]

    print("input shape:",tf.shape(x))
    conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=5, activation = tf.nn.relu, padding = 'same')
    # print ("conv1 shape:",conv1.shape)
    conv1 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=5, activation = tf.nn.relu, padding = 'same')
    # print ("conv1 shape:",conv1.shape)
    conv1 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=5, activation = tf.nn.relu, strides= 2, padding = 'same')
    print ("conv1 shape:",tf.shape(conv1))

    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=5, activation = tf.nn.relu, padding = 'same')
    # print ("conv2 shape:",conv2.shape)
    conv2 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=5, activation = tf.nn.relu, padding = 'same')
    # print ("conv2 shape:",conv2.shape)
    conv2 = tf.layers.conv2d(inputs=conv2, filters=256, kernel_size=3, activation = tf.nn.relu, strides=2, padding = 'same')
    print ("conv2 shape:",tf.shape(conv2))

    conv3 = tf.layers.conv2d(inputs=conv2, filters=256, kernel_size=3, activation = tf.nn.relu, padding = 'same')
    # print ("conv3 shape:",conv3.shape)
    conv3 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=3, activation = tf.nn.relu, padding = 'same')
    # print ("conv3 shape:",conv3.shape)
    conv3 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=3, activation = tf.nn.relu, strides=4, padding = 'same')
    print ("conv3 shape:",tf.shape(conv3))

    # flat1 = tf.layers.flatten(inputs=conv3)
    tensor_shape = conv3.get_shape()
    print(tensor_shape)
    flat1 = tf.reshape(conv3, shape=[-1, tensor_shape[1]*tensor_shape[2]*tensor_shape[3]])
    print("flat ", flat1.get_shape())
    dense1 = tf.layers.dense(inputs=flat1, units=2048, activation=tf.nn.relu)
    print("dense1 ",dense1.get_shape())

    # reshape1 = tf.reshape(dense1, shape = [-1, 2048, 1])
    # print("reshape1", reshape1.shape)

    shortcut = dense1
    res1 = tf.contrib.layers.fully_connected(inputs=dense1,num_outputs=2048,activation_fn=tf.nn.relu)
    # res1 = conv1d(inputs=reshape1, filters=1, kernel_size=3, strides=1, padding='same')
    # print("res1 ", res1.shape)
    # res1 = LeakyReLU()(res1)
    # res1 = batch_normalization(inputs=res1)
    add1 = tf.add(shortcut,res1)
    print("add1 ",add1.shape)
    add1 = LeakyReLU()(add1)
    add1 = tf.layers.batch_normalization(inputs=add1)

    shortcut = add1
    res2 = tf.contrib.layers.fully_connected(inputs=add1,num_outputs=2048,activation_fn=tf.nn.relu)
    # res2 = conv1d(inputs=add1, filters=1, kernel_size=3, strides=1, padding='same')
    # res2 = LeakyReLU()(res2)
    # res2 = batch_normalization(inputs=res2)
    # print("res2 ",res2.shape)
    add2 = tf.add(shortcut,res2)
    print("add2 ",add2.shape)
    add2 = LeakyReLU()(add2)
    add2 = tf.layers.batch_normalization(inputs=add2)

    # add2 = tf.squeeze(add2,[2])
    # print("squeeze", add2.shape)
    weights = tf.contrib.layers.fully_connected(inputs=add2,num_outputs=256,activation_fn=tf.nn.relu)
    # weights = conv1d(inputs=add2, filters=1, kernel_size=3, strides=8, padding='same', activation=tf.nn.relu)
    weights = tf.reshape(weights,[-1,256,1])
    print("weights ", weights.get_shape())

    positions = tf.contrib.layers.fully_connected(inputs=add2,num_outputs=512,activation_fn=tf.nn.sigmoid)
    positions = tf.reshape(positions,[-1,256,2])
    # psize = inputs.shape[1]
    positions = positions*psize
    # positions = conv1d(inputs=add2, filters=2, kernel_size=3, strides=8, padding='same', activation=tf.nn.sigmoid)
    print("positions ",positions.get_shape())

    output = tf.concat([weights,positions], 2)
    print(output.get_shape())
    # return [weights, positions]
    return output
