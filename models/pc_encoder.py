import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions

from .set_transformer import set_transformer

class LatentEncoder(object):
    def __init__(self, hps, name='latent'):
        self.hps = hps
        self.name = name

    def __call__(self, x):
        '''
        x: [B,N,C]
        '''
        B,N,C = tf.shape(x)[0], tf.shape(x)[1], *x.get_shape().as_list()[2:]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = set_transformer(x, self.hps.latent_encoder_hidden, name='set_xformer')
            x = tf.reduce_mean(x, axis=1)
            x = tf.layers.dense(x, self.hps.latent_dim*2, name='d1')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dense(x, self.hps.latent_dim*2, name='d2')
            m, s = x[...,:self.hps.latent_dim], tf.nn.softplus(x[...,self.hps.latent_dim:])
            dist = tfd.Normal(loc=m, scale=s)

        return dist


class SetXformer(object):
    def __init__(self, hps, name='set_xformer'):
        self.hps = hps
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = set_transformer(x, self.hps.set_xformer_hids, name='set_xformer')

        return x