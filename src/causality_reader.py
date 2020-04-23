import os, sys
from glob import glob as glob #glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import ttest_ind, bartlett

###################
# Assumes causality data is given in saved_data/causality
####################

def csv_read(path: str):
	paths = glob(f'{path}/*.csv')
	print(paths)
	dfs = list()
	for path in paths:
		df = pd.read_csv(path, header=0)
		del df[df.columns[0]]
		dfs.append(df)
	return dfs

def summary_analyze(data: pd.DataFrame, visual_pairs: list, condition: tuple):
	if condition: data = data[data[condition[0]] == condition[1]]
	print('shape', data.shape)
	# Mean and std...
	print(data.describe())

	# Covariance matrix
	print(data.corr())
	# plt.matshow(data.corr())
	# plt.show()

	# Joint and marginal
	for pair in visual_pairs:
		sb.jointplot(*pair, data=data)
		plt.show()


def compare(data_old: pd.DataFrame, data_new: pd.DataFrame, t_test_var: str):
	# T test
	_, p = ttest_ind(data_old[t_test_var], data_new[t_test_var], equal_var = False)
	print(f'p val for {t_test_var}: {p}')

	# Variance test
	_, p = bartlett(data_old[t_test_var], data_new[t_test_var])
	print(f'p val variance: {p}')

if __name__ == "__main__":
	os.chdir(sys.path[0])
	datalist = csv_read('saved_data/causality')

	condition = () # ex: ('X', 1)
	data_check, visual_pairs = datalist[0], [('R','S'), ('S','T')]
	summary_analyze(data_check, visual_pairs, condition)

	# data_compares, test_var = (0, 1), 'X'
	# compare(datalist[data_compares[0]], datalist[data_compares[1]], test_var)


