from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
import keras
import numpy as np
from config import ARGS

class DKT(keras.Model):
	def __init__(self, max_skill):
		super(DKT, self).__init__(name='dkt')
		self.__name__ = 'dkt'

		self.max_skill = max_skill
		self.embedding_matrix = np.zeros((self.max_skill+1, ARGS.input_dim))

		self.embedding = Embedding(self.max_skill+1,
									ARGS.input_dim,
									weights=[self.embedding_matrix],
									input_length=ARGS.seq_size)
		self.lstm = LSTM(ARGS.hidden_dim)
		self.dense = Dense(1, activation='sigmoid')

	def call(self, inputs):
		embed = self.embedding(inputs)
		output = self.lstm(embed)
		output = self.dense(output)
		return output
