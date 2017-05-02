# ML-Proj : Deep Structured Energy Based Models for Anomaly Detection
=====================================================================
http://proceedings.mlr.press/v48/zhai16.pdf



Emma Kerinec and Nicolas Derumigny
==================================


Definitions
===========

EBM = density estimatior tool based on a parametrisation of the negative log probability, which is called energy, and then computing the density with a proper normalization

DSEBM = EBM + the energy is learnt by a deep-structured NN (Fully connected for static data, RNN for sequential, CNN for images), output is decided either on energy (DSEBM-e) or reconstruction error (DSEBM-r)


Training by score matching method (intead of maximum likelyhood) by standard stochastic gradient descent (SGD) as a deep denoising autoencoder.






=============================================================================
Use of tensorflow-opencl (https://github.com/benoitsteiner/tensorflow-opencl)
Use of yadlt (http://deep-learning-tensorflow.readthedocs.io)