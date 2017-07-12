# Matching Networks for One Shot Learning
Tensorflow implementation of [Matching Networks for One Shot Learning by Vinyals et al](https://arxiv.org/abs/1606.04080).

## Prerequisites
- Python 2.7+
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)


## Data
- [Omniglot](https://github.com/brendenlake/omniglot)


## Preparation
1. Download and extract omniglot dataset, modify `omniglot_train` and `omniglot_test` in `utils.py` to your location.

2. First time training will generate `omniglot.npy` to the directory. The first 1200 classes are augmented for training so the shape is _(1200, 80, 28, 28, 1)_ which stands for 1200 classes, 20 * 4 90-degree-transform (0, 90, 180, 270), height, width, channel. Rest of the classes are left unchanged (423, 20, 28, 28, 1).

## Train
```bash
python main.py --train
```
Train from a previous checkpoint at epoch X:
```bash
python main.py --train --modelpath=ckpt/model-X
```
Check out tunable hyper-parameters:
```bash
python main.py
```

## Test
```bash
python main.py --eval
```

## Notes
- The model will test the evaluation accuracy after every epoch.
- As the paper indicated, training on Omniglot with FCE does not do any better but I still implemented them (as far as I'm concerned there are no repos that fully implement the FCEs by far).
- The authors did not mentioned the value of time steps K in FCE_f, in the [sited paper](https://arxiv.org/abs/1511.06391), K is tested with 0, 1, 5, 10 as shown in table 1.
- Issues are welcome!

## Resources
- [The paper](https://arxiv.org/abs/1606.04080).
- Referred to [this repo](https://github.com/AntreasAntoniou/MatchingNetworks).
- [Karpathy's note](https://github.com/karpathy/paper-notes/blob/master/matching_networks.md) helps a lot.

