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
SHOW = True

def csv_read(path: str):
	paths = list(reversed(glob(f'{path}/*.csv')))
	print(paths)
	dfs = list()
	for df_path in paths:
		df = pd.read_csv(df_path, header=0)
		del df[df.columns[0]] #del idx
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
		if SHOW: plt.show()


def compare(data_old: pd.DataFrame, data_new: pd.DataFrame, t_test_var: str):
	# T test
	_, p = ttest_ind(data_old[t_test_var], data_new[t_test_var], equal_var = False)
	print(f'p val for {t_test_var} mean: {p}')

	# Variance test
	_, p = bartlett(data_old[t_test_var], data_new[t_test_var])
	print(f'p val {t_test_var} variance: {p}')

if __name__ == "__main__":
	os.chdir(sys.path[0])
	datalist = csv_read('saved_data/causality')

	condition = ()
	# condition = ('S', 1)
	visual_pairs =  [('I','B')]
	summary_analyze(datalist[0], visual_pairs, condition)
	# data_compares, test_var = (0, 1), 'X'
	# data = datalist[0]
	# data_compares = data[data['S'] == 0], data[data['S'] == 1]
	data_compares = datalist[0], datalist[1]
	for var in 'ISBKA':
		compare(*data_compares, var)
	plt.hist(datalist[0]['A'])
	plt.hist(datalist[1]['A'])
	if SHOW: plt.show()
