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

2. First time training will generate `omniglot.npy` to the directory.

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
_UNDER CONSTRUCTION_

## Notes
- FCE (Fully Conditional Embeddings) are not implemented yet but the original result trained on Omniglot did not require that anyway.
- `Data_loader.py` is not done!!
- Issues are welcome!

## Resources
- [The paper](https://arxiv.org/abs/1606.04080).
- Referred to [this repo](https://github.com/AntreasAntoniou/MatchingNetworks).
- [Karpathy's note](https://github.com/karpathy/paper-notes/blob/master/matching_networks.md) helps a lot.

