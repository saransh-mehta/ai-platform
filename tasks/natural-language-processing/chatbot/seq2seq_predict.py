'''
Final Chatbot, question and answer
'''
import tensorflow as tf
import numpy as np
import os


import data
import helper
from seq2seq_model import seq2seq_model

TRAIN_DATA_PERCENT = 0.8
TEST_DATA_PERCENT = 0.1
VAL_DATA_PERCENT = 0.1

# loading all data
metadata, idxQ, idxA = data.load_data(os.getcwd() + '/')

MAX_LEN_X = metadata['limit']['maxq']
MAX_LEN_Y = metadata['limit']['maxa']
VOCAB_SIZE_ENCODER = len(metadata['idx2w'])
VOCAB_SIZE_DECODER = VOCAB_SIZE_ENCODER # since same language
EMBED_DIMS = 200
HIDDEN_UNITS = 200
NUMBER_OF_LAYERS = 3
LEARNING_RATE = 0.001
DROPOUT = 0.5

BATCH_SIZE = 32
TEST_EPOCHS = 1
SAVED_MODEL_DIR = 'saved_model_seq2seq'
# shuffling data
idxQ, idxA = helper.shuffle_data(idxQ, idxA)

# splitting data into train, test, validation
trainX,trainY,testX,testY,valX,valY = helper.split_data(idxQ,idxA, TRAIN_DATA_PERCENT,
														TEST_DATA_PERCENT, VAL_DATA_PERCENT)

# creating model class object
model = seq2seq_model(vocabSizeEncoder = VOCAB_SIZE_ENCODER, vocabSizeDecoder = VOCAB_SIZE_DECODER,
					maxLenX = MAX_LEN_X, maxLenY = MAX_LEN_Y, embedDims = EMBED_DIMS,
					numLayers = NUMBER_OF_LAYERS, hiddenUnits = HIDDEN_UNITS,
					lr = LEARNING_RATE)

# re- building tensorflow graph

model.build_model_graph()

# creating saver object to restore
saver = tf.train.Saver()

# test batch generator object
batchGenTest = helper.get_next_batch(BATCH_SIZE, testX, testY)

# restoring

with tf.Session() as sess:
	saver.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_DIR))

	wish = 'y'
	while wish.lower() == 'y':
		# generating batch
		batchX, batchY = batchGenTest.__next__()
		print(np.array(batchX).shape)
		#print(batchX)
		# generate feed_dict
		feedDict = helper.create_prediction_feed_dict(model, batchX)

		# prediction 

		decoderTestOut = sess.run(model.decoderOutTest, feed_dict=feedDict)
		decoderTestOut = np.array(decoderTestOut)
		print(decoderTestOut.shape)

		# this output is in Time Major format, transposing it to batch Major
		decoderTestOut = decoderTestOut.transpose([1, 0, 2])
		print(decoderTestOut.shape)

		# choosing the word with max softmax probability

		finalOut = np.argmax(decoderTestOut, axis = 2)

		# checking if the output is in expected shape
		print(finalOut.shape)
		assert(finalOut.shape == (BATCH_SIZE, MAX_LEN_Y))

		for eachInp, eachOut in zip(batchX, finalOut):

			#convert back to text
			question = ' '.join([ metadata['idx2w'][index] for index in eachInp])

			answer = ' '.join([ metadata['idx2w'][index] for index in eachOut ])

			print('\nquestion: ', question)
			print('answer: ', answer)

		wish = str(input('Do you wish to predict for next test batch (y/n): '))



