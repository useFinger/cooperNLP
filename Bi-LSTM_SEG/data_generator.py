#encoding=utf-8
# 导入数据
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BatchGenerator(object):
	""" Construct a Data generator. The input X, y should be ndarray or list like type.
	
	Example:
		Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
		Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
		X = Data_train.X
		y = Data_train.y
		or:
		X_batch, y_batch = Data_train.next_batch(batch_size)
	 """ 
	
	def __init__(self, X, y, shuffle=False):
		if type(X) != np.ndarray:
			X = np.asarray(X)
		if type(y) != np.ndarray:
			y = np.asarray(y)
		self._X = X
		self._y = y
		self._epochs_completed = 0
		self._index_in_epoch = 0
		self._number_examples = self._X.shape[0]
		self._shuffle = shuffle
		if self._shuffle:
			new_index = np.random.permutation(self._number_examples)
			self._X = self._X[new_index]
			self._y = self._y[new_index]
				
	@property
	def X(self):
		return self._X
	
	@property
	def y(self):
		return self._y
	
	@property
	def num_examples(self):
		return self._number_examples
	
	@property
	def epochs_completed(self):
		return self._epochs_completed
	
	def next_batch(self, batch_size):
		""" Return the next 'batch_size' examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._number_examples:
			# finished epoch
			self._epochs_completed += 1
			# Shuffle the data 
			if self._shuffle:
				new_index = np.random.permutation(self._number_examples)
				self._X = self._X[new_index]
				self._y = self._y[new_index]
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._number_examples
		end = self._index_in_epoch
		return self._X[start:end], self._y[start:end]

def data_gen():
	with open('data/data.pkl', 'rb') as inp:
		X = pickle.load(inp)
		y = pickle.load(inp)
		word2id = pickle.load(inp)
		id2word = pickle.load(inp)
		tag2id = pickle.load(inp)
		id2tag = pickle.load(inp)
		labels = pickle.load(inp)

# 划分测试集/训练集/验证集

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.2, random_state=42)
	print ('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

	print ('Creating the data generator ...')
	data_train = BatchGenerator(X_train, y_train, shuffle=True)
	data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
	data_test = BatchGenerator(X_test, y_test, shuffle=False)
	print ('Finished creating the data generator.')

	return data_train, data_valid, data_test, word2id, id2word, tag2id, id2tag, labels