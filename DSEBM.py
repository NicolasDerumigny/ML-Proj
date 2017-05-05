import tensorflow as tf
import numpy as np
import math
import os

## Original code from https://gist.github.com/blackecho/db85fab069bd2d6fb3e7#file-rbm_after_refactor-py

## Utils

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
    	#TODO : remove visible unit type, make it as an array
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

        self.models_dir, self.data_dir, self.summary_dir = self._create_data_directories()
        self.model_path = self.models_dir + self.model_name

        self.W = [None]*len(layer_shape)
        self.bias = [None]*len(layer_shape)
        self.global_bias = None

        self.w_update = [None]*len(layer_shape)
        self.bias_update = [None]*len(layer_shape)
        self.global_bias_update = None
        self.encode = None

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
            self.tf_saver.save(self.tf_session, self.model_path)

    def _initialize_tf_utilities_and_ops(self, restore_previous_model):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        """

        self.tf_merged_summaries = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

        self.tf_summary_writer = tf.train.SummaryWriter(self.summary_dir, self.tf_session.graph_def)

    #TODO : train by score matching method
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

        updates = [self.w_update, self.bias_update, self.global_bias_update]

        self.tf_session.run(updates, feed_dict=self._create_feed_dict(train_set))


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
        :param data: training/validation set
        :return: dictionary(self.input_data: data, self.hrand: random_uniform, self.vrand: random_uniform)
        """

        return {
            self.input_data: data
        }

    def _build_model(self):

        """ Build the FCDSEBM model in TensorFlow.
        #TODO
        :return: self
        """

        self.input_data = self._create_placeholders()
        self.W, self.bias_, self.global_bias_ = self._create_variables()

        hprobs0, hstates0, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(self.input_data) 
        positive = self.compute_positive_association(self.input_data, hprobs0, hstates0)

        nn_input = vprobs


        negative = tf.matmul(tf.transpose(vprobs), hprobs1)

        self.encode = hprobs1  # encoded data, used by the transform method

        for i in range (len(self.layer_shape)):
        	self.w_update[i] = self.W[i].assign_add(self.learning_rate * (positive[i] - negative[i]))
       		self.bias_update[i] = self.bias[i].assign_add(self.learning_rate * tf.reduce_mean(hprobs0[i] - hprobs1[i], 0))
        self.global_bias_update = self.global_bias.assign_add(self.learning_rate * tf.reduce_mean(self.input_data - vprobs, 0))

        self.loss_function = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - vprobs)))
        _ = tf.scalar_summary("cost", self.loss_function)

    def _create_placeholders(self):

        """ Create the TensorFlow placeholders for the model.
        :return: tuple(input(shape(None, num_visible)))
        """

        x = tf.placeholder('float', [None, self.layer_shape[0]] name='x-input')

        return x

    def _create_variables(self):

        """ Create the TensorFlow variables for the model.
        :return: tuple(weights(shape(num_visible, num_hidden),
                       hidden bias(shape(num_hidden)),
                       visible bias(shape(num_visible)))
        """

        for i in range (1, len(layer_shape)):
        	W[i] = tf.Variable(tf.random_normal((self.layer_shape[i-1], self.layer_shape[i]), mean=0.0, stddev=0.01), name='weights'+str(i))
        	bias[i] = tf.Variable(tf.zeros([self.layer_shape[i]]), name='bias'+str(i))
        global_bias = tf.Variable(tf.zeros([self.layer_shape[0]]), name='global-bias')

        return W, bias, global_bias

    def gibbs_sampling_step(self, inir):

        """ Performs one step of gibbs sampling.
        :param init: activations of the visible units
        :return: tuple(hidden probs, hidden states, visible probs,
                       new hidden probs, new hidden states)
        """

        hprobs, hstates = self.sample_from_next_units(visible)
        vprobs = self.sample_from_prec_units(hprobs)
        hprobs1, hstates1 = self.sample_from_next_units(vprobs)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def sample_from_next_units(self, init):

        """ Sample the hidden units from the visible units.
        This is the Positive phase of the Contrastive Divergence algorithm.
        :param visible: activations of the visible units
        :return: tuple(hidden probabilities, hidden binary states)
        """

        hprobs=[tf.nn.sigmoid(tf.matmul(init, self.W[1]) + self.bias[0])]
        for i in range(1, len(layer_shape)):
        	hprobs = tf.nn.sigmoid(tf.matmul(self.W[i-1], self.W[i]) + self.bias[i-1])
        
        hstates = sample_prob(hprobs, self.hrand)

        return hprobs, hstates

    def sample_from_prec_units(self, hidden):

        """ Sample the visible units from the hidden units.
        This is the Negative phase of the Contrastive Divergence algorithm.
        :param hidden: activations of the hidden units
        :return: visible probabilities
        """

        visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.vis_bias_

        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal((1, self.num_visible), mean=visible_activation, stddev=self.stddev)

        else:
            vprobs = None

        return vprobs

    def compute_positive_association(self, visible, hidden_probs, hidden_states):

        """ Compute positive associations between visible and hidden units.
        :param visible: visible units
        :param hidden_probs: hidden units probabilities
        :param hidden_states: hidden units states
        :return: positive association = dot(visible.T, hidden)
        """

        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None

        return positive

    def _create_data_directories(self):

        """ Create the three directories for storing respectively the models,
        the data generated by training and the TensorFlow's summaries.
        :return: tuple of strings(models_dir, data_dir, summary_dir)
        """

        self.main_dir = self.main_dir + '/' if self.main_dir[-1] != '/' else self.main_dir

        models_dir = config.models_dir + self.main_dir
        data_dir = config.data_dir + self.main_dir
        summary_dir = config.summary_dir + self.main_dir

        for d in [models_dir, data_dir, summary_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)

        return models_dir, data_dir, summary_dir

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

    def load_model(self, shape, gibbs_sampling_steps, model_path):

        """ Load a trained model from disk. The shape of the model
        (num_visible, num_hidden) and the number of gibbs sampling steps
        must be known in order to restore the model.
        :param shape: tuple(num_visible, num_hidden)
        :param gibbs_sampling_steps:
        :param model_path:
        :return: self
        """

        self.num_visible, self.num_hidden = shape[0], shape[1]
        self.gibbs_sampling_steps = gibbs_sampling_steps

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

            return {
                'W': [k.eval() for k in self.W],
                'bias': [k.eval() for k in self.bias],
                'global_bias_': self.global_bias_.eval()
            }