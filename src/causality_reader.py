import os, sys
from glob import glob as glog #glob

import numpy as np

####
# Assumes causality data is given in saved_data/causality
####

def csv_read(path: str):
	for csvpath in glob(f'{path}/*.csv'):
		raise NotImplementedError
	# return data
if __name__ = "__main__":
	os.chdir(sys.path[0])
	data = csv_read('saved_data/causality')

