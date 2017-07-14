from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import glob

import scipy.misc
import numpy as np

from tqdm import tqdm

def read_omniglot():
    """Read omniglot dataset, save them to a single npy file"""
    omniglot_train = '/home/one-shot-dataset/omniglot/python/images_background' 
    omniglot_eval = '/home/one-shot-dataset/omniglot/python/images_evaluation' 
    
    data = []
    for r in [omniglot_train, omniglot_eval]:
        classes = glob.glob(r + '/*')
        for cls in tqdm(classes):
            alphabets = glob.glob(cls + '/*')
            for a in alphabets:
                characters = glob.glob(a + '/*')
                raws = []
                for ch in characters: # 20 iters
                    raw = scipy.misc.imread(ch)
                    raw = scipy.misc.imresize(raw, (28, 28))
                    for dg in [0, 90, 180, 270]: # augmentation
                        raw_rot = scipy.misc.imrotate(raw, dg)
                        raw_rot = raw_rot[:, :, np.newaxis] # (28, 28, 1)
                        raw_rot = raw_rot.astype(np.float32) / 255.
                        raws.append(raw_rot)
                data.append(np.asarray(raws))
    np.save('omniglot.npy', np.asarray(data))

            
class Data_loader():

    def __init__(self, batch_size, n_way=5, k_shot=1, train_mode=True):
        if not os.path.exists('omniglot.npy'):
           read_omniglot()

        self.batch_size = batch_size
        self.n_way = n_way  # 5 or 20, how many classes the model has to select from
        self.k_shot = k_shot # 1 or 5, how many times the model sees the example
    
        omniglot = np.load('omniglot.npy')
        #omniglot = np.load('data_zergylord.npy')
        #omniglot = np.reshape(omniglot, [-1, 20, 28, 28, 1])
        np.random.shuffle(omniglot)
        assert omniglot.dtype == np.float32
        assert omniglot.max() == 1.0
        assert omniglot.min() == 0.0

        if train_mode:
            self.images = omniglot[:1200, :20, :, :, :]
            self.num_classes = self.images.shape[0]
            self.num_samples = self.images.shape[1]
        else:
            self.images = omniglot[1200:, :20, :, :, :]
            self.num_classes = self.images.shape[0]
            self.num_samples = self.images.shape[1]

        self.iters = self.num_classes

    def next_batch(self):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []
        for _ in xrange(self.batch_size):
            x_set = []
            y_set = []
            x = []
            y = []
            classes = np.random.permutation(self.num_classes)[:self.n_way]
            target_class = np.random.randint(self.n_way)
            for i, c in enumerate(classes):
                samples = np.random.permutation(self.num_samples)[:self.k_shot+1]
                for s in samples[:-1]:
                    x_set.append(self.images[c][s])
                    y_set.append(i)

                if i == target_class:
                    x_hat_batch.append(self.images[c][samples[-1]])
                    y_hat_batch.append(i)

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

        return np.asarray(x_set_batch).astype(np.float32), np.asarray(y_set_batch).astype(np.int32), np.asarray(x_hat_batch).astype(np.float32), np.asarray(y_hat_batch).astype(np.int32)
