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

def evaluate(args):

    loader = Data_loader(args.bsize, args.n_way, args.k_shot, train_mode=False)
    model = Matching_Nets(args.lr, args.n_way, args.k_shot, args.use_fce, args.bsize)

    model.build(model.support_set_image_ph, model.support_set_label_ph, model.example_image_ph)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    if args.modelpath is not None:
        print ('Using model: {}'.format(args.modelpath))
        saver.restore(sess, args.modelpath)
    else:
        latest_ckpt = tf.train.latest_checkpoint('log')
        print ('Using latest: {}'.format(latest_ckpt))

    correct = 0
    for _ in xrange(loader.iters):
        x_set, y_set, x_hat, y_hat = loader.next_batch()
        feed_dict = {model.support_set_image_ph: x_set,
                     model.support_set_label_ph: y_set,
                     model.example_image_ph: x_hat}
        logits, prediction = sess.run([model.logits, model.pred], feed_dict=feed_dict)
        correct += np.sum(np.equal(prediction, y_hat))

    print ('Evaluation accuracy: %.2f%%' % (correct * 100 / (loader.iters * args.bsize)))

def train(args):

    train_loader = Data_loader(args.bsize, args.n_way, args.k_shot)
    eval_loader = Data_loader(args.bsize, args.n_way, args.k_shot, train_mode=False)
    model = Matching_Nets(args.lr, args.n_way, args.k_shot, args.use_fce, args.bsize)

    model.build(model.support_set_image_ph, model.support_set_label_ph, model.example_image_ph)
    model.loss(model.example_label_ph)
    train_op = model.train()

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if args.modelpath is not None:
        print ('From model: {}'.format(args.modelpath))
        saver.restore(sess, args.modelpath)

    print ('Start training')
    print ('batch size: %d, ep: %d, iter: %d, initial lr: %.3f' % (args.bsize, args.ep, train_loader.iters, args.lr))

    for ep in xrange(args.ep):
        # start training
        correct = 0
        for step in xrange(train_loader.iters):
            x_set, y_set, x_hat, y_hat = train_loader.next_batch()
            feed_dict = {model.support_set_image_ph: x_set,
                         model.support_set_label_ph: y_set,
                         model.example_image_ph: x_hat,
                         model.example_label_ph: y_hat}
            logits, prediction, loss, _ = sess.run([model.logits, model.pred, model.loss_op, train_op], feed_dict=feed_dict)
            correct += np.sum(np.equal(prediction, y_hat))
            
            if step % 100 == 0:
                print ('ep: %3d, step: %3d, loss: %.3f' % (ep+1, step, loss))

        print ('  Training accuracy: %.2f%%' % (correct * 100 / (train_loader.iters * args.bsize)))
        checkpoint_path = os.path.join('log', 'matchnet.ckpt')
        saver.save(sess, checkpoint_path, global_step=ep+1)

        # training for one epoch done, evaluate on test set
        correct = 0
        for step in xrange(eval_loader.iters + 1):
            x_set, y_set, x_hat, y_hat = eval_loader.next_batch()
            feed_dict = {model.support_set_image_ph: x_set,
                         model.support_set_label_ph: y_set,
                         model.example_image_ph: x_hat}
            logits, prediction = sess.run([model.logits, model.pred], feed_dict=feed_dict)
            correct += np.sum(np.equal(prediction, y_hat))

        print ('Evaluation accuracy: %.2f%%' % (correct * 100 / ((eval_loader.iters + 1) * args.bsize)))

    print ('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--eval', action='store_true', help='set this to evaluate.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=100, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=32, help='batch size.')
    parser.add_argument('--n-way', metavar='', type=int, default=5, help='number of classes.')
    parser.add_argument('--k-shot', metavar='', type=int, default=1, help='number of chances the model see.')
    parser.add_argument('--use-fce', metavar='', type=bool, default=True, help='use fully conditional embedding or not.')
    parser.add_argument('--modelpath', metavar='', type=str, default=None, help='trained tensorflow model path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if args.eval:
        evaluate(args)
    if not args.train and not args.eval:
        parser.print_help()
