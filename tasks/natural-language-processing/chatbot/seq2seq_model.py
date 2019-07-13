'''
Sequence to Sequence Model for chatbot
'''
import tensorflow as tf
import numpy as np
import os

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

class seq2seq_model:

	def __init__(self, vocabSizeEncoder, vocabSizeDecoder, maxLenX, maxLenY,
				embedDims, numLayers, hiddenUnits, lr):

		self.vocabSizeEncoder = vocabSizeEncoder
		self.vocabSizeDecoder = vocabSizeDecoder 
		# in case we want to use it for language translation, vocabs differ
		self.maxLenX = maxLenX
		self.maxLenY = maxLenY

		# max number of time steps for encoder and decoder, from our data
		# it is 20 for encoder,20 for decoder

		self.embedDims = embedDims
		self.numLayers = numLayers
		# number of layers of LSTM we want to stack in the model
		self.hiddenUnits = hiddenUnits
		self.lr = lr # learning rate for optimizer

	def build_model_graph(self):
		tf.reset_default_graph()

		with tf.variable_scope('placeholders') as scope:
			self.encoderInp = [ tf.placeholder(shape = (None,), dtype=tf.int32,
								name = '_'.join(['encIp', str(tStep)])) for tStep in range(self.maxLenX)]

			# we have made a list of placeholders, i.e one placeholder for each time-step
			# this list of placeholder is haveing max time steps (20) number of placeholders
			# each placeholder will hold the batch, that's y shape (None,) for that time steps

			self.decoderTarget = [ tf.placeholder(shape = (None, ), dtype = tf.int32,
									name = '_'.join(['decTar', str(tStep)])) for tStep in range(self.maxLenY)]

			# to hold decoder targets similarly

			# Now decoder inp will be actually one time step behind the decoder output and just
			# the first time step of decoder inp will contain <'GO'> token

			self.decoderInp = [ tf.zeros_like(self.encoderInp[0], dtype = tf.int32, name ='go_')]+ self.decoderTarget[:-1]
			# 'go' is simply represented by 0, which is '_'

			# NOTE:- decoderInp is not a placeholder, it will be created shifting encoderInp placeholder,

			self.keepProb = tf.placeholder(shape = (None), dtype = tf.float32, name='dropout')
			# for adding droupout in tf, we need a tensor of probability to be kept

		with tf.variable_scope('Cell') as scope:

			cell = tf.nn.rnn_cell.DropoutWrapper(
						tf.nn.rnn_cell.BasicLSTMCell(self.hiddenUnits,state_is_tuple = True),
						output_keep_prob = self.keepProb)
			# this dropout wrapper adds dropout to LSTM Cell

			# stacking multiple layers of LSTM cell
			stackedLSTM = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(self.numLayers)],state_is_tuple=True)
			# we could have used different number of hidden units in the cells

			'''
			Tensorflow has two major functions which can be used for implementing Seq2Seq in tensorflow

			outputs, states = basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
			and
			outputs, states = embedding_rnn_seq2seq(
			encoder_inputs, decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols,
			embedding_size, output_projection=None, feed_previous=False)

			Second functions handles the embedding part also,
			As decoder uses the out produced at t-1 to feed along with input for time step t,
			during training, since the decoder will produce wrong result at t-1, we have to still use 
			the target value provided through decoder_input_placeholder for training.
			This is done by keeping feed_previous = False in the above function

			When it is true, tf only uses the first element of decoder inp to initialize decoder and
			then feeds consecutive outputs.
			'''
		with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE) as scope:
			#scope.reuse_variables()
			self.decoderOutTrain, self.decoderStateTrain = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.encoderInp,
																self.decoderInp, stackedLSTM,
																self.vocabSizeEncoder, self.vocabSizeDecoder,
																self.embedDims)

			# we will make another copy of the above encoder-decoder with feed_previous=True,
			# so at the time of predictio we will use this copy to make predictions.
			# Now as tf is graph based and adds everything to graph, but we dont want to add 
			# 2 seq2seq encoder-decoder, hence, putting reuse = True instead of adding another
			# node to graph, will just reuse the variable already declared in scope.
			# hence we get our copy.
			self.decoderOutTest, self.decoderStateTest = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.encoderInp,
																self.decoderInp, stackedLSTM,
																self.vocabSizeEncoder, self.vocabSizeDecoder,
																self.embedDims, feed_previous=True)
		with tf.variable_scope('loss') as scope:

			# tf.contrib has a fuction for loss in seq2se models,

			lossWeights = [ tf.ones_like(label, dtype=tf.float32) for label in self.decoderTarget ]
			self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decoderOutTrain, self.decoderTarget,
														lossWeights, self.vocabSizeDecoder)
			
			self.lossVal = tf.contrib.legacy_seq2seq.sequence_loss(self.decoderOutTest, self.decoderTarget,
														lossWeights, self.vocabSizeDecoder)

		with tf.variable_scope('optimizer') as scope:

			optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
			self.train = optimizer.minimize(self.loss)











