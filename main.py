import sys
import time
import torch
import random
import torch.nn as nn
from config import ARGS
from models import *
from utils import *
from callbacks import *

if __name__ == '__main__':
	data_path = ARGS.base_path + "/" + ARGS.dataset_name + "/" + ARGS.dataset_name
	train_path = data_path + "_train.csv"
	valid_path = data_path + "_valid" + str(ARGS.valid_index) + ".csv"
	test_path = data_path + "_test.csv"
	timing = time.asctime()
	logger = get_logger(input_file="logs/{0}-{1}.log".format(ARGS.model_type, timing).replace(':', '_'))
	logging_args(logger, ARGS)

	# Loading datasets
	train_students, train_correctness, train_problems, train_skills = read_data(ARGS.seq_size, train_path, shuffle=True)
	valid_students, valid_correctness, valid_problems, valid_skills = read_data(ARGS.seq_size, valid_path, shuffle=False)
	test_students, test_correctness, test_problems, test_skills = read_data(ARGS.seq_size, test_path, shuffle=False)

	train_X, train_y = pad_seq(train_students, train_correctness, ARGS.seq_size, train_skills)
	valid_X, valid_y = pad_seq(valid_students, valid_correctness, ARGS.seq_size, valid_skills)
	test_X, test_y = pad_seq(test_students, test_correctness, ARGS.seq_size, test_skills)

	# Logging basic information
	max_skill = max(train_skills, valid_skills, test_skills)
	print(f'Dataset size: train: {train_X.shape} valid: {valid_X.shape} test: {test_X.shape}')
	print(f'Skill Count: train: {train_skills} valid: {valid_skills} test: {test_skills}')
	print(f'Max Sequence Length: train: {train_problems} valid: {valid_problems} test: {test_problems}')

	logger.info(f'Dataset size: train: {train_X.shape} valid: {valid_X.shape} test: {test_X.shape}')
	logger.info(f'Skill Count: train: {train_skills} valid: {valid_skills} test: {test_skills}')
	logger.info(f'Max Sequence Length: train: {train_problems} valid: {valid_problems} test: {test_problems}')

	# Training and testing
	model = DKT(ARGS.seq_size)
	ARGS.weight_path = ARGS.weight_path + "/" + ARGS.model_type

	if ARGS.only_save_best:
		callbacks = get_checkpoints(ARGS.weight_path + "_{epoch:02d}-{val_acc:.2f}_best.hdf5", verbose=0, only_save_best=True, mode="max", perios=1)
	else:
		callbacks = LossHistory()

	model = trainer(model, callbacks, logger,
					train_X, train_y,
					valid_X, valid_y,
					test_X, test_y,
		 			threshold=0.01)

	if not ARGS.only_save_best:
		model.save_weights(ARGS.weight_path+"_last.h5")
		callbacks.loss_log(logger, type="epoch")
		callbacks.loss_plot("epoch")
