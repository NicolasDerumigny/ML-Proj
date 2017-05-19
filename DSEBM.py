import tensorflow as tf
import numpy as np
import math
import os
from random import randint, seed

## Original code from https://gist.github.com/blackecho/db85fab069bd2d6fb3e7#file-rbm_after_refactor-py


models_dir = 'models/'  # dir to save/restore models
data_dir = 'data/'  # directory to store algorithm data
summary_dir = 'logs/'  # directory to store tensorflow summaries


def g(x):
	''' Soft plus function '''
	return tf.log(1 + tf.exp(x))

def sample_prob(probs, rand):
	""" Takes a tensor of probabilities (as from a sigmoidal activation)
	and samples from all the distributions
	:param probs: tensor of probabilities
	:param rand: tensor (of the same shape as probs) of random values
	:return : binary sample of probabilities
	"""
	return tf.nn.relu(tf.sign(probs - rand))


class FC_DSEBM(object):

	""" Fully Connected Deep-Structured Energy Based Model
	implementation using TensorFlow.
	The interface of the class is sklearn-like.
	"""

	def __init__(self, layer_shape, visible_unit_type='bin', main_dir='rbm', model_name='rbm_model',
				learning_rate=0.01, num_epochs=10, stddev=0.1, verbose=0):
		"""
		:param layer_shape: array of the number of neuron per layer
		:param visible_unit_type: type of the visible units (binary or gaussian)
		:param main_dir: main directory to put the models, data and summary directories
		:param model_name: name of the model, used to save data
		:param learning_rate: optional, default 0.01
		:param num_epochs: optional, default 10
		:param stddev: optional, default 0.1. Ignored if visible_unit_type is not 'gauss'
		:param verbose: level of verbosity. optional, default 0
		"""

		self.layer_shape = layer_shape
		self.visible_unit_type = visible_unit_type
		self.main_dir = main_dir
		self.model_name = model_name
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.stddev = stddev
		self.verbose = verbose

		self.W = [None]*len(layer_shape)
		self.bias = [None]*len(layer_shape)
		self.global_bias = None

		self.train = None

		self.loss_function = None

		self.input_data = None
		self.validation_size = None

		self.tf_merged_summaries = None
		self.tf_summary_writer = None
		self.tf_session = None
		self.tf_saver = None

	def fit(self, train_set, validation_set=None, restore_previous_model=False):

		""" Fit the model to the training data.
		:param train_set: training set
		:param validation_set: validation set. optional, default None
		:param restore_previous_model:
					if true, a previous trained model
					with the same name of this model is restored from disk to continue training.
		:return: self
		"""

		if validation_set is not None:
			self.validation_size = validation_set.shape[0]

		self._build_model()

		with tf.Session() as self.tf_session:

			self._initialize_tf_utilities_and_ops(restore_previous_model)
			self._train_model(train_set, validation_set)


	def reconstruct(self, data):
		"""Reconstruct data according to the model.
		Parameters
		----------
		data : array_like, shape (n_samples, n_features)
			Data to transform.
		graph : tf.Graph, optional (default = None)
			Tensorflow Graph object
		Returns
		-------
		array_like, transformed data
		"""

		with tf.Session() as self.tf_session:
			feed = {self.input_data: data}
			return self.reconstruction.eval(feed)


	def score(self, data, data_ref, graph=None):
		"""Compute the reconstruction loss over the test set.
		Parameters
		----------
		data : array_like
			Data to reconstruct.
		data_ref : array_like
			Reference data.
		Returns
		-------
		float: Mean error.
		"""

		with tf.Session() as self.tf_session:
			feed = {
				self.input_data: data,
				self.input_labels: data_ref,
				self.keep_prob: 1
			}
			return self.cost.eval(feed)



	def _initialize_tf_utilities_and_ops(self, restore_previous_model):

		""" Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
		Restore a previously trained model if the flag restore_previous_model is true.
		"""

		self.tf_merged_summaries = tf.summary.merge_all()
		init_op = tf.global_variables_initializer()
		self.tf_saver = tf.train.Saver()

		self.tf_session.run(init_op)

		if restore_previous_model:
			self.tf_saver.restore(self.tf_session, self.model_path)


	def _train_model(self, train_set, validation_set):

		""" Train the model.
		:param train_set: training set
		:param validation_set: validation set. optional, default None
		:return: self
		"""

		for i in range(self.num_epochs):
			self._run_train_step(train_set)

			if validation_set is not None:
				self._run_validation_error_and_summaries(i, validation_set)

	def _run_train_step(self, train_set):

		""" Run a training step.
		:param train_set: training set
		:return: self
		"""

		self.tf_session.run(self.train, feed_dict=self._create_feed_dict(train_set))


	def _run_validation_error_and_summaries(self, epoch, validation_set):

		""" Run the summaries and error computation on the validation set.
		:param epoch: current epoch
		:param validation_set: validation data
		:return: self
		"""

		result = self.tf_session.run([self.tf_merged_summaries, self.loss_function],
									 feed_dict=self._create_feed_dict(validation_set))

		summary_str = result[0]
		err = result[1]

		self.tf_summary_writer.add_summary(summary_str, 1)

		if self.verbose == 1:
			print("Validation cost at step %s: %s" % (epoch, err))

	def _create_feed_dict(self, data):

		""" Create the dictionary of data to feed to TensorFlow's session during training.
		return only one parameter because of the SGD algorithm
		:param data: training/validation set
		:return: dictionary(self.input_data: data, self.hrand: random_uniform, self.vrand: random_uniform)
		"""

		return {
			self.input_data: np.reshape(data[randint(0,len(data)-1)], (len(data[0]), 1)) #SGD algorithm
		}
    
	def _build_model(self):
		""" Build the FCDSEBM model in TensorFlow.
		:return: self
		"""

		self.input_data = self._create_placeholders()
		self.W, self.bias, self.global_bias = self._create_variables()

		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

		h=[None]*len(self.layer_shape)
		h[0]=self.input_data

		for i in range (1, len(self.layer_shape)):
			h[i]=g(tf.matmul(tf.transpose(self.W[i]), h[i-1]) + self.bias[i])

		result = tf.zeros((self.layer_shape[-1],1))+1
		for i in range (len(self.layer_shape)-2, -1, -1):
			result = tf.matmul(self.W[i+1],\
					 		   tf.sigmoid(tf.matmul(tf.transpose(self.W[i+1]),h[i]) + self.bias[i+1]))\
					 * tf.matmul(self.W[i+1], result)

		result = result + self.global_bias


		self.loss_function = tf.reduce_sum(tf.square(self.input_data - result))
		self.train = optimizer.minimize(result)

	def _create_placeholders(self):

		""" Create the TensorFlow placeholders for the model.
		:return: tuple(input(shape(None, num_visible)))
		"""

		x = tf.placeholder('float', [self.layer_shape[0], 1], name='x-input')

		return x

	def _create_variables(self):

		""" Create the TensorFlow variables for the model.
		:return: tuple(weights(shape(num_visible, num_hidden),
					   hidden bias(shape(num_hidden)),
					   visible bias(shape(num_visible)))
		"""
		W=[None]*len(self.layer_shape)
		bias=[None]*len(self.layer_shape)
		for i in range (1, len(self.layer_shape)):
			W[i] = tf.Variable(tf.random_normal((self.layer_shape[i-1], self.layer_shape[i]), mean=0.0, stddev=0.01), name='weights'+str(i))
			bias[i] = tf.Variable(tf.zeros((self.layer_shape[i],1)), name='bias'+str(i))
		global_bias = tf.Variable(tf.zeros((self.layer_shape[0],1)), name='global-bias')

		return W, bias, global_bias

	def transform(self, data, name='train', save=False):

		""" Transform data according to the model.
		:type data: array_like
		:param data: Data to transform
		:type name: string, default 'train'
		:param name: Identifier for the data that is being encoded
		:type save: boolean, default 'False'
		:param save: If true, save data to disk
		:return: transformed data
		"""

		with tf.Session() as self.tf_session:

			self.tf_saver.restore(self.tf_session, self.model_path)

			encoded_data = self.encode.eval(self._create_feed_dict(data))

			if save:
				np.save(self.data_dir + self.model_name + '-' + name, encoded_data)

			return encoded_data

	def load_model(self, shape, model_path):

		""" Load a trained model from disk. The shape of the model
		(num_visible, num_hidden) and the number of gibbs sampling steps
		must be known in order to restore the model.
		:param shape: tuple(num_visible, num_hidden)
		:param model_path:
		:return: self
		"""

		self.num_visible, self.num_hidden = shape[0], shape[1]

		self._build_model()

		init_op = tf.initialize_all_variables()
		self.tf_saver = tf.train.Saver()

		with tf.Session() as self.tf_session:

			self.tf_session.run(init_op)
			self.tf_saver.restore(self.tf_session, model_path)

	def get_model_parameters(self):

		""" Return the model parameters in the form of numpy arrays.
		:return: model parameters
		"""

		with tf.Session() as self.tf_session:

			self.tf_saver.restore(self.tf_session, self.model_path)

			return [{
				'W': [k.eval() for k in self.W[i]],
				'bias': [k.eval() for k in self.bias[i]],
				'global_bias': self.global_bias.eval()
			} for i in range(layer_shape)]