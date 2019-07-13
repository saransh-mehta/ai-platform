'''
Training seq2seq chatbot
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
BATCH_SIZE = 32
DROPOUT = 0.5
EPOCHS = 1000
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model_seq2seq', 'model_seq2seq_')
LOG_FILE = 'terminal_out_seq2seq.txt'

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

# building tensorflow graph

model.build_model_graph()

#training part

# function to make feed_dict, as there's a list of placeholders

def create_feed_dict(model, placeX, placeY, placeDropout):
	feedDict = {}
	# adding placeholder for droupout value
	feedDict[model.keepProb] = placeDropout

	# adding values for maxLen (i.e 20) placeholders for encoder Inp
	feedDict.update({model.encoderInp[tStep] : placeX[tStep] for tStep in range(model.maxLenX)})
	'''
	We are working with time Major, thats why we transposed the result of batchGen from
	(batchSize, maxTimeSteps) to (maxTimeSteps, batchSize) and are using maxTimeSteps number
	of placeholders, so now we can access the inputs to be fed into placeholder
	just by accessing their first axis, placeX[tStep].

	model.encoderInp[tStep] will provide the placeholder for which placeX[tStep] has to be fed as
	data. I hope tf supports this kinda addition in feed_dict, else I explicitly mentioned name 
	of placeholder
	'''
	feedDict.update({model.decoderTarget[tStep] : placeY[tStep] for tStep in range(model.maxLenY)})

	# update method in dict, adds content of 2nd dict into first

	return feedDict

# creating iter object from batch generator function
# as yield in batch function will return an iterator, we will use __next__()
# to produce next batches

batchGenTrain = helper.get_next_batch(BATCH_SIZE, trainX, trainY)
#print(batchGenTrain)
batchGenVal = helper.get_next_batch(BATCH_SIZE, valX, valY)


# saver object 
saver = tf.train.Saver()
# initializer object
init = tf.global_variables_initializer()
# creating session
sess = tf.Session()
# initializing
sess.run(init)

# train

for i in range(EPOCHS):

	# generating batch
	batchX, batchY = batchGenTrain.__next__()
	#print(batchX)
	# generate feed_dict
	feedDict = create_feed_dict(model, batchX, batchY, DROPOUT)

	_, lossTrain = sess.run([model.train, model.loss], feed_dict = feedDict)

	if i % 5 == 0:
		# save model
		saver.save(sess, MODEL_SAVE_PATH, global_step = i)

		# validation 

		batchValX, batchValY = batchGenVal.__next__()
		feedDictVal = create_feed_dict(model, batchValX, batchValY, 1.0)

		lossVal, decoderValOut = sess.run([model.lossVal, model.decoderOutTest], feed_dict=feedDictVal)

		# this output is in Time Major format, transposing it to batch Major
		decoderValOut = np.array(decoderValOut).transpose([1, 0, 2])

		# writing outputs into log file
		toPrint = str(i) + " train Loss: {}".format(lossTrain) + " val Loss: {}".format(lossVal)
		print(toPrint)
		with open(LOG_FILE, 'a') as f:
			f.write(toPrint)







