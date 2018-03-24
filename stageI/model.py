from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import misc.custom_ops
from misc.custom_ops import leaky_rectify
from misc.config import cfg
import tensorflow.contrib.slim as slim


class CondGAN(object):
    def __init__(self, image_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = image_shape
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.image_shape = image_shape
        self.s = image_shape[0]
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

        # Since D is only used during training, we build a template
        # for safe reuse the variables during computing loss for fake/real/wrong images
        # We do not do this for G,
        # because batch_norm needs different options for training and testing
        if cfg.GAN.NETWORK_TYPE == "default":
            with tf.variable_scope("d_net"):
                self.d_encode_img_template = self.d_encode_image()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        elif cfg.GAN.NETWORK_TYPE == "simple":
            with tf.variable_scope("d_net"):
                self.d_encode_img_template = self.d_encode_image_simple()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        else:
            raise NotImplementedError

    # g-net
    def generate_condition(self, c_var):
        conditions =\
            (pt.wrap(c_var).
             flatten().
             custom_fully_connected(self.ef_dim * 2).
             apply(leaky_rectify, leakiness=0.2))
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]

    def generator(self, z_var):
        with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, is_training=True):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                biases_initializer=None):
                net10 = slim.fully_connected(z_var, 4 * 4 * self.gf_dim * 8, activation_fn=None, scope='net10',
                                             weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                             normalizer_fn=slim.batch_norm)
                net10 = tf.reshape(net10, [-1, 4, 4, self.gf_dim * 8])
                net11 = slim.conv2d(net10, self.gf_dim * 2, 1, scope='net11_1')
                net11 = slim.conv2d(net11, self.gf_dim * 2, 3, scope='net11_2')
                net11 = slim.conv2d(net11, self.gf_dim * 8, 3, activation_fn=None, scope='net11_3')
                net1 = tf.nn.relu(net10 + net11)
                net20 = tf.image.resize_nearest_neighbor(net1, [8, 8])
                net20 = slim.conv2d(net20, self.gf_dim * 4, 3, activation_fn=None, scope='net20')
                net21 = slim.conv2d(net20, self.gf_dim, 1, scope='net21_1')
                net21 = slim.conv2d(net21, self.gf_dim, 3, scope='net21_2')
                net21 = slim.conv2d(net21, self.gf_dim * 4, 3, activation_fn=None, scope='net21_3')
                net2 = tf.nn.relu(net20 + net21)
                net = tf.image.resize_nearest_neighbor(net2, [16, 16])
                net = slim.conv2d(net, self.gf_dim * 2, 3, scope='net3_1')
                net = tf.image.resize_nearest_neighbor(net, [32, 32])
                net = slim.conv2d(net, self.gf_dim, 3, scope='net3_2')
                net = tf.image.resize_nearest_neighbor(net, [64, 64])
                net = slim.conv2d(net, 3, 3, activation_fn=tf.tanh, normalizer_fn=None, scope='output')
                return net

    def generator_simple(self, z_var):
        output_tensor =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             # custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             # custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             # custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             # custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor

    def get_generator(self, z_var):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(z_var)
        elif cfg.GAN.NETWORK_TYPE == "simple":
            return self.generator_simple(z_var)
        else:
            raise NotImplementedError

    # d-net
    def context_embedding(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template

    def d_encode_image(self):
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1

    def d_encode_image_simple(self):
        template = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2))

        return template

    def discriminator(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))

        return template

    def get_discriminator(self, x_var, c_var):
        with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, is_training=True):
            with slim.arg_scope([slim.conv2d], activation_fn=leaky_relu, normalizer_fn=slim.batch_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                biases_initializer=None):
                net10 = slim.conv2d(x_var, self.df_dim, 4, stride=2, normalizer_fn=None, scope='net10_1')
                net10 = slim.conv2d(net10, self.df_dim * 2, 4, stride=2, scope='net10_2')
                net10 = slim.conv2d(net10, self.df_dim * 4, 4, stride=2, activation_fn=None, scope='net10_3')
                net10 = slim.conv2d(net10, self.df_dim * 8, 4, stride=2, activation_fn=None, scope='net10_4')
                net11 = slim.conv2d(net10, self.df_dim * 2, 1, scope='net11_1')
                net11 = slim.conv2d(net11, self.df_dim * 2, 3, scope='net11_2')
                net11 = slim.conv2d(net11, self.df_dim * 8, 3, scope='net11_3')
                net1 = leaky_relu(net10 + net11)

                context = slim.fully_connected(c_var, self.ef_dim, activation_fn=leaky_relu,
                                               weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc')
                context = tf.expand_dims(tf.expand_dims(context, 1), 1)
                context = tf.tile(context, [1, 4, 4, 1])
                net = tf.concat(3, [net1, context])

                net = slim.conv2d(net, self.df_dim * 8, 1, scope='net2')
                net = slim.conv2d(net, 1, 4, padding='VALID', activation_fn=None, normalizer_fn=None, scope='output')
                net = tf.squeeze(net, [1, 2])
                return net

def leaky_relu(x, leakiness=0.2):
  assert leakiness <= 1
  ret = tf.maximum(x, leakiness * x)
  # import ipdb; ipdb.set_trace()
  return ret
