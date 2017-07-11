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
    omniglot_train = '/home/one-shot-dataset/omniglot/python/images_background/' 
    omniglot_eval = '/home/one-shot-dataset/omniglot/python/images_evaluation/' 
    
    data = []
    for r in [omniglot_train, omniglot_eval]:
        classes = sorted(glob.glob(r + '*'))
        for i, c in enumerate(tqdm(classes)):
            alphabets = sorted(glob.glob(c + '/*'))
            for j, a in enumerate(alphabets): # a = [0, num_of_chars]
                characters = sorted(glob.glob(a + '/*'))
                raws = []
                for k, ch in enumerate(characters): # k = [0, 19]
                    raw = scipy.misc.imread(ch)
                    raw = scipy.misc.imresize(raw, (28, 28))
                    raw = raw[:, :, np.newaxis] # (28, 28, 1)
                    raws.append(raw)
                raws = np.asarray(raws)
                data.append(raws)
    data = np.asarray(data)
    np.save('omniglot.npy', data)
            
class Data_loader():

    def __init__(self, batch_size, n_way=5, k_shot=1):
        if not os.path.exists('omniglot.npy'):
            read_omniglot()

        omniglot_images = np.load('omniglot.npy')
        omniglot_labels = np.arange(0, omniglot_images.shape[0])
        self.batch_size = batch_size
        
        self.train_images = omniglot_images[:1200] # (1200, 20, 28, 28, 1)
        self.train_labels = omniglot_labels[:1200] # (1200,)
        self.train_classes = self.train_images.shape[0]
        
        self.test_images = omniglot_images[1200:]  # (423, 20, 28, 28, 1)
        self.test_labels = omniglot_labels[1200:]  # (423,)
        self.test_classes = self.test_images.shape[0]

        self.n_way = n_way  # 5 or 20, how many classes the model has to select from
        self.k_shot = k_shot # 1 or 5, how many times the model sees the example
        self.num_samples = omniglot_images.shape[1]
        self.iters = self.train_classes // self.n_way // self.k_shot * self.num_samples // self.batch_size

        self.reset_episode()

    def single_example(self):
        """Return a single example"""
        # build the support set x
        x_set = []
        y_set = []
        for n in xrange(self.n_way): # from n-classes, select k-samples
            for k in xrange(self.k_shot):
                x_set.append(self.train_images[self.n_ptr + n][self.k_ptr + k])
                y_set.append(n) #self.train_labels[self.n_ptr + n])

        # the target x_hat
        if self.k_ptr + 1 >= self.num_samples:
            select_k = 0
        else:
            select_k = self.k_ptr + 1
        x_hat = self.train_images[self.n_ptr][select_k]
        y_hat = 0 #self.train_labels[self.n_ptr]

        return np.asarray(x_set), np.asarray(y_set), x_hat, y_hat

    def next_batch(self):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []
        for b in xrange(self.batch_size):
            x_set, y_set, x_hat, y_hat = self.single_example()
            x_set_batch.append(x_set)
            y_set_batch.append(y_set)
            x_hat_batch.append(x_hat)
            y_hat_batch.append(y_hat)

            num_classes = self.train_classes
            if self.n_ptr + self.n_way < num_classes:
                # iterate through all the classes first
                self.n_ptr += self.n_way
            else:
                # all classes traversed, reset class pointer and increment sample pointer
                self.n_ptr = 0
                self.k_ptr += self.k_shot
            
            if self.k_ptr + self.k_shot <= self.num_samples:
                # same as above but let the class pointer decide k_ptr's fate
                pass
            else:
                # all samples and classes traversed, reset everything
                self.reset_episode()

        return np.asarray(x_set_batch), np.asarray(y_set_batch), np.asarray(x_hat_batch), np.asarray(y_hat_batch)

    def reset_episode(self):
        self.n_ptr = 0
        self.k_ptr = 0

        p = np.random.permutation(self.train_images.shape[0]) # shuffle x and y together
        self.train_images = self.train_images[p]
        self.train_labels = self.train_labels[p]

#dl = Data_loader(32)
#a, b, c, d = dl.next_batch()
#pdb.set_trace()
#print ('done')
