import pdb
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
rnn = tf.contrib.rnn

class Matching_Nets():

    def __init__(self, lr, n_way, k_shot, use_fce, batch_size=32):
        self.lr = lr
        self.n_way = n_way
        self.k_shot = k_shot
        self.use_fce = use_fce
        self.batch_size = batch_size
        self.processing_steps = 10

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

    def fce_g(self, encoded_x_i):
        """the fully conditional embedding function g
        This is a bi-directional LSTM, g(x_i, S) = h_i(->) + h_i(<-) + g'(x_i) where g' is the image encoder
        For omniglot, this is not used.

        encoded_x_i: g'(x_i) in the equation.   length n * k list of (batch_size ,64)
        """
        fw_cell = rnn.BasicLSTMCell(32) # 32 is half of 64 (output from cnn)
        bw_cell = rnn.BasicLSTMCell(32)
        outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(fw_cell, bw_cell, encoded_x_i, dtype=tf.float32)

        return tf.add(tf.stack(encoded_x_i), tf.stack(outputs))

    def fce_f(self, encoded_x, g_embedding):
        """the fully conditional embedding function f
        This is just a vanilla LSTM with attention where the input at each time step is constant and the hidden state
        is a function of previous hidden state but also a concatenated readout vector.
        For omniglot, this is not used.

        encoded_x: f'(x_hat) in equation (3) in paper appendix A.1.     (batch_size, 64)
        g_embedding: g(x_i) in equation (5), (6) in paper appendix A.1. (n * k, batch_size, 64)
        """
        cell = rnn.BasicLSTMCell(64)
        prev_state = cell.zero_state(self.batch_size, tf.float32) # state[0] is c, state[1] is h

        for step in xrange(self.processing_steps):
            output, state = cell(encoded_x, prev_state) # output: (batch_size, 64)
            
            h_k = tf.add(output, encoded_x) # (batch_size, 64)

            content_based_attention = tf.nn.softmax(tf.multiply(prev_state[1], g_embedding))    # (n * k, batch_size, 64)
            r_k = tf.reduce_sum(tf.multiply(content_based_attention, g_embedding), axis=0)      # (batch_size, 64)

            prev_state = rnn.LSTMStateTuple(state[0], tf.add(h_k, r_k))

        return output

    def cosine_similarity(self, target, support_set):
        """the c() function that calculate the cosine similarity between (embedded) support set and (embedded) target
        
        note: the author uses one-sided cosine similarity as zergylord said in his repo (zergylord/oneshot)
        """
        #target_normed = tf.nn.l2_normalize(target, 1) # (batch_size, 64)
        target_normed = target
        sup_similarity = []
        for i in tf.unstack(support_set):
            i_normed = tf.nn.l2_normalize(i, 1) # (batch_size, 64)
            similarity = tf.matmul(tf.expand_dims(target_normed, 1), tf.expand_dims(i_normed, 2)) # (batch_size, )
            sup_similarity.append(similarity)

        return tf.squeeze(tf.stack(sup_similarity, axis=1)) # (batch_size, n * k)

    def build(self, support_set_image, support_set_label, image):
        """the main graph of matching networks"""
        image_encoded = self.image_encoder(image)   # (batch_size, 64)
        support_set_image_encoded = [self.image_encoder(i) for i in tf.unstack(support_set_image, axis=1)]

        if self.use_fce:
            g_embedding = self.fce_g(support_set_image_encoded)     # (n * k, batch_size, 64)
            f_embedding = self.fce_f(image_encoded, g_embedding)    # (batch_size, 64)
        else:
            g_embedding = tf.stack(support_set_image_encoded)       # (n * k, batch_size, 64)
            f_embedding = image_encoded                             # (batch_size, 64)

        # c(f(x_hat), g(x_i))
        embeddings_similarity = self.cosine_similarity(f_embedding, g_embedding) # (batch_size, n * k)

        # compute softmax on similarity to get a(x_hat, x_i)
        attention = tf.nn.softmax(embeddings_similarity)

        # \hat{y} = \sum_{i=1}^{k} a(\hat{x}, x_i)y_i
        y_hat = tf.matmul(tf.expand_dims(attention, 1), tf.one_hot(support_set_label, self.n_way))
        self.logits = tf.squeeze(y_hat)   # (batch_size, n)

        self.pred = tf.argmax(self.logits, 1)

    def loss(self, label):
        self.loss_op = tf.losses.sparse_softmax_cross_entropy(label, self.logits)

    def train(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss_op)
