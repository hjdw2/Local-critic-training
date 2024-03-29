# Local-critic-training
## Introduction

This repository is the Tensorflow code for the paper [Local Critic Training for Model-Parallel Learning of Deep Neural Networks]. 

These codes are examples for CIFAR-10 with ResNet-101.

For any question or suggestions, feel free to contact hjlee92@yonsei.ac.kr

## Dependencies

* Python 3.5.2 (Anaconda)
* Tensorflow 1.2
* CUDA 9.0


## Run

Clone and cd into the repo directory and extract CIFAR-10 data to the same directory, 

run for LCT_n1: 
```
python LCT_n1.py 
``` 

## Citation 

```latex
@ARTICLE{9358982,
  author={H. {Lee} and C. -J. {Hsieh} and J. -S. {Lee}},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Local Critic Training for Model-Parallel Learning of Deep Neural Networks}, 
  year={2022},
  volume={33},
  number={9},
  pages={4424-4436},
  doi={10.1109/TNNLS.2021.3057380}}
```


