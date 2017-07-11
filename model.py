import pdb
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

class Matching_Nets():

    def __init__(self, lr, n_way, k_shot):
        self.lr = lr
        self.n_way = n_way
        self.k_shot = k_shot
        self.support_set_image_ph = tf.placeholder(tf.float32, [None, n_way * k_shot, 28, 28, 1])
        self.support_set_label_ph = tf.placeholder(tf.int32, [None, n_way * k_shot, ])

        self.example_image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.example_label_ph = tf.placeholder(tf.int32, [None, ])

    def image_encoder(self, image):
        """the embedding function for image (potentially f = g)
        For omniglot it's a simple 4 layer ConvNet, for mini-imagenet it's VGG or Inception
        """
        with slim.arg_scope([slim.conv2d], num_outputs=64, kernel_size=3, normalizer_fn=slim.batch_norm):
            net = slim.conv2d(image)
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net)
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net)
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net)
            net = slim.max_pool2d(net, [2, 2])
        return tf.reshape(net, [-1, 1 * 1 * 64])

    def fce_g(self):
        """the fully conditional embedding function g
        This is a bi-directional LSTM over the examples, g(x_i, S) = h_i(->) + h_i(<-) + g'(x_i) where g' is the image encoder
        For omniglot, this is not used.
        """
        pass

    def fce_f(self):
        """the fully conditional embedding function f
        This is really just a vanilla LSTM with attention where the input at each time step is constant and the hidden state
        is a function of previous hidden state but also a concatenated readout vector.
        For omniglot, this is not used.
        """
        pass

    def cosine_similarity(self, support_set_image, image):
        """the c() function that calculate the cosine similarity between the support set and the image"""
        image_normed = tf.nn.l2_normalize(image, 1) # (batch_size, 64)
        sup_similarity = []
        for i in tf.unstack(support_set_image, axis=1):
            sup_i_normed = tf.nn.l2_normalize(i, 1) # (batch_size, 64)
            similarity = tf.reduce_sum(tf.multiply(image_normed, sup_i_normed), axis=1) # (batch_size, )
            sup_similarity.append(similarity)

        return tf.stack(sup_similarity, axis=1)

    def build(self, support_set_image, support_set_label, image):
        """the main graph of matching networks"""
        image_encoded = self.image_encoder(image)   # (batch_size, 64)
        
        support_set_image_encoded = [self.image_encoder(i) for i in tf.unstack(support_set_image, axis=1)]
        support_set_image_encoded = tf.stack(support_set_image_encoded, axis=1) # (batch_size, n * k, 64)

        support_set_image_similarity = self.cosine_similarity(support_set_image_encoded, image_encoded) # (batch_size, n * k)

        # compute softmax on similarity to get a(x_hat, x_i)
        attention = tf.nn.softmax(support_set_image_similarity)   # (batch_size, n * k)

        # \hat{y} = \sum_{i=1}^{k} a(\hat{x}, x_i)y_i
        y_hat = tf.matmul(tf.expand_dims(attention, 1), tf.one_hot(support_set_label, self.n_way * self.k_shot))
        self.logits = tf.squeeze(y_hat)   # (batch_size, n * k)

        self.pred = tf.argmax(self.logits, 1)

    def loss(self, label):
        self.loss_op = tf.losses.sparse_softmax_cross_entropy(label, self.logits)

    def train(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss_op)
