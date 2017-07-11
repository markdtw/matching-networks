from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import argparse

import numpy as np
import tensorflow as tf

from model import Matching_Nets
from utils import Data_loader

def train(args):

    loader = Data_loader(args.bsize)
    model = Matching_Nets(args.lr, args.n_way, args.k_shot)

    model.build(model.support_set_image_ph, model.support_set_label_ph, model.example_image_ph)
    model.loss(model.example_label_ph)
    train_op = model.train()

    tf.summary.scalar('loss', model.loss_op)
    merged_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    writer = tf.summary.FileWriter('log', sess.graph)
    sess.run(tf.global_variables_initializer())

    if args.modelpath is not None:
        print ('From model: {}'.format(args.modelpath))
        saver.restore(sess, args.modelpath)

    print ('Start training')
    print ('batch size: %d, ep: %d, iter: %d, initial lr: %.4f' % (args.bsize, args.ep, loader.iters, args.lr))

    for ep in xrange(args.ep):
        for step in xrange(loader.iters):
            x_set, y_set, x_hat, y_hat = loader.next_batch()
            feed_dict = {model.support_set_image_ph: x_set,
                         model.support_set_label_ph: y_set,
                         model.example_image_ph: x_hat,
                         model.example_label_ph: y_hat}
            logits, prediction, loss, _, summary = sess.run([model.logits, model.pred, model.loss_op, train_op, merged_op], feed_dict=feed_dict)
            #pdb.set_trace()
            
            writer.add_summary(summary, ep * loader.iters + step)
            if step % 10 == 0:
                print ('ep: %2d, step: %2d, loss: %.4f' %
                        (ep+1, step, loss))

        checkpoint_path = os.path.join('log', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=ep+1)

    print ('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=20, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=32, help='batch size.')
    parser.add_argument('--n-way', metavar='', type=int, default=5, help='number of classes.')
    parser.add_argument('--k-shot', metavar='', type=int, default=1, help='number of chances the model see.')
    parser.add_argument('--modelpath', metavar='', type=str, default=None, help='trained tensorflow model path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if not args.train:
        parser.print_help()
