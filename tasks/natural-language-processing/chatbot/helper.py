'''
Few helper functions for generating trainingbatches and splitting data into test, train, validation
as dataset is too big
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

SEED = 2


def shuffle_data(qArray, aArray, seed = SEED):
	return shuffle(qArray, aArray, random_state = seed)

def split_data(qArray, aArray, trainSize, testSize, valSize):
	length = len(qArray)

	size = [trainSize, testSize, valSize]
	newLengths = [int(len(qArray)*percent) for percent in size]

	trainX, trainY = qArray[:newLengths[0]], aArray[:newLengths[0]]

	testX, testY = qArray[newLengths[0]:newLengths[0]+newLengths[1]], aArray[newLengths[0]:newLengths[0]+newLengths[1]]

	valX, valY = qArray[newLengths[0]+newLengths[1]:], aArray[newLengths[0]+newLengths[1]:]

	return trainX, trainY, testX, testY, valX, valY

def get_next_batch(batchSize, x, y):

	while True:
		batchIndices = np.random.choice( ( list(np.arange(len(x))) ), batchSize )

		X = np.array([x[index] for index in batchIndices]).T
		Y = np.array([y[index] for index in batchIndices]).T

		'''
		IMPORTANT: here use of .T is very significant, as we have decided to work in
		time-major = True, first axis should represent the time step and second batch size
		we have make placeholders, feed_dict creation according.
		Error in this could lead to irrelevant training
		'''
		yield X, Y

def create_prediction_feed_dict(model, x):
	feedDict = {}
	# adding placeholder for droupout value
	feedDict[model.keepProb] = 1.0

	# adding values for maxLen (i.e 20) placeholders for encoder Inp
	feedDict.update({model.encoderInp[tStep] : x[tStep] for tStep in range(model.maxLenX)})

	return feedDict






