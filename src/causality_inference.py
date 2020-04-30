import os, sys
from itertools import combinations
from glob import glob as glob #glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import ttest_ind, bartlett

from helpers import pearsonr_ci, corrfunc, meanfunc

###################
# Assumes causality data is given in saved_data/causality
####################
sb.set(style="white")
SHOW = True

def csv_read(path: str):
	paths = list(sorted(glob(f'{path}/*.csv')))
	print(paths)
	dfs = list()
	for df_path in paths:
		df = pd.read_csv(df_path, header=0)
		del df[df.columns[0]] #del idx
		dfs.append(df)
	return dfs

def summary_analyze(data: pd.DataFrame, visual_pairs: list = [], condition: tuple = ()):
	if condition: data = data[data[condition[0]] == condition[1]]
	print('shape', data.shape)
	# Mean and std...
	print(data.describe())

	# Covariance matrix
	for pair in combinations(data.columns, r=2):
		print(f"{pair[0]} and {pair[1]}")
		pearsonr_ci(data[pair[0]], data[pair[1]])
	# print(data.corr())
	# plt.matshow(data.corr())
	# plt.show()

	# Joint and marginal
	g = sb.PairGrid(data, palette=["red"])
	g.map_diag(sb.distplot, kde=False, bins=10)
	g.map_diag(meanfunc)
	# g.map_upper(sb.kdeplot, cmap="Blues_d")

	g.map_lower(plt.scatter, s=10)
	g.map_lower(corrfunc)
	plt.show()

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

	summary_analyze(datalist[0], condition = ('S', 1))
	# summary_analyze(datalist[0])
	# summary_analyze(datalist[1])

	# summary_analyze(datalist[2])

	# summary_analyze(datalist[3])

	# condition = ('S', 1)
	# summary_analyze(datalist[3])
	# summary_analyze(datalist[3], condition=('S',1))
	# summary_analyze(datalist[3])
	# summary_analyze(datalist[2])

	# data = datalist[3]
	# data_compares = data[data['S'] == 1], data
	# data_compares = datalist[0], datalist[3]
	# for var in 'APKI':
		# compare(*data_compares, var)


	# plt.boxplot( datalist[0]['K'], )
	# plt.show()
	# plt.boxplot( datalist[3]['K'] )
	# plt.show()
