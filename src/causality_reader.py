import os, sys
from glob import glob as glog #glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats impor ttest_ind

####
# Assumes causality data is given in saved_data/causality
####

def csv_read(path: str):
	return pd.read_csv(path) for path in glob(f'{path}/*.csv')

def summary_analyze(data: pd.Dataframe, visual_pairs: list):
	# Mean and std...
	print(data.describe())

	# Covariance matrix
	plt.matshow(data.corr())
	plt.show()
	plt.cfg()

	# Joint and marginal
	for pair in visual_pairs: sb.jointplot(*pair, data=data)

def compare(data_old: pd.Dataframe, data_new: pd.Dataframe, t_test_var: str):
	# T test
	_, p = ttest_ind(data_old[t_test_var], data_new[t_test_var], equal_var = False)
	print(f'p val for {t_test_var}: {p}')



if __name__ == "__main__":
	os.chdir(sys.path[0])
	datalist = csv_read('saved_data/causality')
	data_check = datalist[0]
	summary_analyze(data_check, visual_pairs = [('X','Y'), ('Z','F')])



