# Progressive Neural Architecture Search with Binarized Neural Networks

**This project combines the architecture search strategy from [Progressive Neural Architecture Search][1] with the search space of Binarized Neural Networks [Binarized Neural Networks][2].** 


If you find this code useful, please consider citing the original work by authors:

```
@article{liu2017progressive,
  title={Progressive neural architecture search},
  author={Liu, Chenxi and Zoph, Barret and Shlens, Jonathon and Hua, Wei and Li, Li-Jia and Fei-Fei, Li and Yuille, Alan and Huang, Jonathan and Murphy, Kevin},
  journal={arXiv preprint arXiv:1712.00559},
  year={2017}
}
```

```
@inproceedings{hubara2016binarized,
  title={Binarized neural networks},
  author={Hubara, Itay and Courbariaux, Matthieu and Soudry, Daniel and El-Yaniv, Ran and Bengio, Yoshua},
  booktitle={Advances in neural information processing systems},
  pages={4107--4115},
  year={2016}
}
```

Introduction
------------
Neural Architecture Search is a sub-field of AutoML which has recently gained popularity for generating state-of-the-art architectures on various tasks of Image Processing and Natural Language Processing. 


Progressive Neural Architecture Search searches through the space in a sequential fashion starting with simplest models and increasing the complexity as it proceeds. It learns a surrogate reward predictor implemented as a RNN to reduce the overhead of training every proposed architecture. 

<p align="center">
  <img src="https://imgur.com/JyGBZyh.png">
</p>


Binarized Neural Networks with binary weights and activations at run-time drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations which substantially improve power-efficiency. Both the weights and the activations are constrained to either +1 or -1. 

Binarization function used in the experiment is deterministic binary-tanh which is placed in this [file][3]. 


Setup Dependencies
-----
The recommended version for running the experiments is Python3, however it works fine with Python2.

1. Follow the installation guide on [Tensorflow Homepage][4] for installing Tensorflow-GPU or Tensorflow-CPU. 
2. Follow instructions outlined on [Keras Homepage][5] for installing Keras.










[1]:https://arxiv.org/abs/1712.00559
[2]:https://arxiv.org/abs/1602.02830
[3]:https://github.com/yashkant/PNAS-Binarized-Neural-Networks/blob/master/binarized/binary_ops.py
[4]:https://www.tensorflow.org/install/
[5]:https://keras.io/#installation
