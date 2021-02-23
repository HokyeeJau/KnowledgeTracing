import sys
import time
import random
import torch
import torch.nn as nn
import keras
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.utils import shuffle
from keras.models import load_model
from config import ARGS
import logging
import os
import csv
import matplotlib.pyplot as plt

def read_data(seq_size, path, shuffle=False):
	'''
	Load data in a fixed length. Abandon the former sequence of the over-size sequence
	'''
	ds_path = path

	rows = []
	max_skill_num = 0
	max_num_problems = seq_size

	with open(ds_path, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			rows.append(row)

	index = 0
	print("The Number of Rows is " + str(len(rows)))
	tuple_rows = []
	interaction = []
	correctness = []
	while(index < len(rows) - 1):
		problems_num = int(rows[index][0])
		tmp_max_skill = max(map(int, rows[index+1]))
		max_skill_num = max(max_skill_num, tmp_max_skill)

		if problems_num <= 2:
			index += 3
		else:
			if problems_num > max_num_problems:
				inte = list(map(int, rows[index+1][-max_num_problems:]))
				corr = list(map(int, rows[index+2][-max_num_problems:]))
				count = problems_num // max_num_problems
				iii = 0
				while(iii <= count):
					if iii != count:
						inte = list(map(int, rows[index+1][iii*max_num_problems : (iii+1)*max_num_problems]))
						corr = list(map(int, rows[index+2][iii*max_num_problems : (iii+1)*max_num_problems]))
					elif problems_num - iii*max_num_problems > 2:
						inte = list(map(int, rows[index+1][iii*max_num_problems : (iii+1)*max_num_problems]))
						corr = list(map(int, rows[index+2][iii*max_num_problems : (iii+1)*max_num_problems]))
					else: break
					interaction.append(inte)
					correctness.append(corr)
					iii += 1
				index += 3
			else:
				inte = list(map(int, rows[index+1]))
				corr = list(map(int, rows[index+2]))
				interaction.append(inte)
				correctness.append(corr)
				index += 3

	if shuffle:
		random.shuffle(tuple_rows)
	print("The number of Students is {}.".format(len(interaction)))
	return interaction, correctness, max_num_problems, max_skill_num

def pad_seq(interaction, correctness, max_steps, MAX_SKILLS):
	inlist = []
	colist = []
	for i in range(len(interaction)):
		assert len(interaction[i]) == len(correctness[i]), "length error"
		inlist.append([0]*(max_steps-len(interaction[i]))+interaction[i])
		colist.append([0]*(max_steps-len(correctness[i]))+correctness[i])

	inarr = np.array(inlist)
	coarr = np.array(colist)
	# inarr = inarr + MAX_SKILLS*(coarr == 0)
	return inarr, coarr[:, -1]

def get_logger(verbosity=1, input_file=None, name="torchlog"):
	level_dict = {0: logging.DEBUG,
				  1: logging.INFO,
				  2: logging.WARNING}
	formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")

	logger = logging.getLogger(name)
	logger.setLevel(level_dict[verbosity])

	log_dir = os.path.dirname(input_file)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	fh = logging.FileHandler(input_file, mode='w')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	return logger

def get_args():
	params = parser.parse_known_args()[0]

	# Set down device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if params.gpu != 'none':
		os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu

	torch.manual_seed(params.random_seed)
	torch.cuda.manual_seed(params.random_seed)
	torch.cuda.manual_seed_all(params.random_seed)
	random.seed(params.random_seed)

	if torch.cuda.is_available():
		params.device = 'cuda'
		params.gpu = list(range(len(params.gpu.split(','))))
		if params.gpu is not None:
			torch.cuda.set_device(params.gpu[0])
	params.dataset_name = ds_list[params.ds]
	params.weight_path = f'weight/{params.dataset_name}/{params.model}/'
	os.makedirs(params.weight_path, exist_ok=True)
	return params

def logging_args(logger, args):
	dilimeter = '-'*100
	logger.info('\n'.join([dilimeter,
					  *['{}|{}'.format(key.upper(), value) for key, value in args._get_kwargs()],
					  dilimeter]))

def load_model(path):
	return load_model(path)

class ClassificationReport:
	def __init__(self, test_X, test_y, logger, model, threshold):
		self._test = test_X
		self._result = test_y
		self._logger = logger
		self._model = model
		self._threshold = threshold
		self._name = model.__name__

		logger.info(f'Test shape: {self._test.shape}')
		logger.info(f'test threshold: {self._threshold}')

		y_predict = model.predict(self._test, verbose=1)
		y_predict = (y_predict > self._threshold).astype(int)
		self.y_true = np.reshape(self._result, [-1])
		self.y_pred = np.reshape(y_predict, [-1])


	def __report__(self):
		labels = [0, 1]
		target_names = ['label 0', 'label 1']
		self._logger.info("\n"+classification_report(self.y_true,
									self.y_pred,
									labels=labels,
									target_names=target_names,
									digits=4))
		print(classification_report(self.y_true,
									self.y_pred,
									labels=labels,
									target_names=target_names,
									digits=4))
	def __roc__(self):
		fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred)
		self._logger.info(f"AUC: {auc(fpr, tpr)}")
		print(f"AUC: {auc(fpr, tpr)}")
		path = './roc_curve/'
		timing = time.asctime()
		plt.figure(dpi=600)
		plt.plot(fpr,tpr,marker = 'o')
		plt.xlabel('fpr')
		plt.ylabel('tpr')
		plt.grid(True)
		plt.title(timing)
		plt.savefig(f"{path}{self._name}-{auc(fpr, tpr)}.png")

def trainer(model,
				callback_list,
				logger,
				train_X, train_y,
				valid_X, valid_y,
				test_X, test_y,
	 			threshold):
	opt = keras.optimizers.Adam(learning_rate=ARGS.lr)
	model.compile(optimizer=opt, loss='binary_crossentropy')

	print("-----Train-----")
	# fit network
	hist = model.fit(train_X,
					train_y,
					epochs=ARGS.num_epochs,
					batch_size=ARGS.train_batch,
					validation_data=(valid_X, valid_y),
					verbose=1,
					shuffle=True,
					callbacks=[callback_list])
	model.summary()
	logger.info(model.summary())
	if type(callback_list).__name__ == 'Callback':
		callback_list.loss_plot('epoch')
		callback_list.loss_plot('batch')

	report = ClassificationReport(test_X, test_y, logger, model, threshold)
	report.__report__()
	report.__roc__()
	return model
