# tracksuite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research library for training, testing and evaluating state-of-the-art Visual Object Tracking algorithms. It is intended to be easy-to-use and also provide a testbed for fast experimentation and research with Visual Object Tracking. 

## Content
+ [Models](#models)
+ [Datasets](#datasets)
+ [Pretrained Models](#pretrained-models)
+ [Dependencies](#dependencies)
+ [Roadmap](#roadmap)

## Models
### GOTURN
GOTURN: Generic Object Tracking Using Regression Networks based on the paper: [Learning to Track at 100 FPS with Deep Regression Networks](http://davheld.github.io/GOTURN/GOTURN.html) by [David Held](http://davheld.github.io/), [Sebastian Thrun](http://robots.stanford.edu/) and [Silvio Savarese](http://cvgl.stanford.edu/silvio/)

#### Implementation details
1. The model is trained on only the ALOV300++ video dataset i.e. it hasn't been trained along with the Imagenet Detection dataset as recommended in the paper.
2. The motion smoothness model (for augmenting the training set) has not been implemented. 
3. We have also used the Adam optimizer instead of the Sochastic Gradient Descent with Momentum as recommended in the paper as this gives better results. A learning rate of `1e-6` was used.
4. With this current setup the model achieves `~62%` accuracy (based on IoU only) on a held-out validation set.

#### To-Do
+ Experiment with training on Imagenet Video dataset (Like in SiamFC)
+ Implement motion smoothness which is easy to use with torch `transforms`

## Datasets
+ [ALOV300++ Dataset](http://alov300pp.joomlafree.it/)

## Pretrained Models
+ [GOTURN](https://drive.google.com/file/d/1-qzu3knrE7KYYW0Djkb_YNnvdluz2KEW/view?usp=sharing)

## Dependencies
+ pytorch >=0.4.0 
+ torchvision torchvision >=0.2.0
+ scipy
+ opencv3
+ tensorboardX
+ tdqm

## Roadmap
Here are next few networks to be implemented:
+ SiamFC
+ CFNet
+ SiamRPN

Datasets to implement:
+ OTB
+ VOT
+ Imagenet Video
+ Imagenet Detection
