import numpy as np
import json

from copy import copy

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import accuracy_score

def dataprepare(bins=4, split=0.6):
	with np.load('src/saved_data/corona.npz') as data:
		X, y = data['X'], data['y']
	with open('src/saved_data/names.json', 'r') as infile:
		namedata = json.load(infile)

	country_names,feature_names = namedata['countries'], namedata['features']

	y =  KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile').fit_transform(y.reshape(-1, 1))
	X_last = OneHotEncoder(categories='auto', sparse=False).fit_transform(X[:, -1].reshape(-1, 1))
	X = StandardScaler().fit_transform(X[:, :-1])

	X = np.hstack( (X, X_last) )

	scrampled_idcs = np.random.permutation(X.shape[0])
	cutoff = int(split*X.shape[0])
	pool_idcs, test_idcs = list(scrampled_idcs[:cutoff]), scrampled_idcs[cutoff:]

	#There needs to be examples of each class in both test and train
	assert len(np.unique(y[test_idcs])) == bins and len(np.unique(y[pool_idcs])) == bins

	return X, y.flatten(), pool_idcs, test_idcs, country_names, feature_names


random = lambda probs: 0
least_conf = lambda probs: np.argsort(1 - probs.max(1))[0]
margin = lambda probs: np.argsort(probs.max(1)-probs.min(1))[-1]

def entropy(probs):
	probsum = np.zeros(probs.shape[0])
	for i in range(1,probs.shape[1]):
		probsum += probs[:,i]*np.log(probs[:,i])
	return np.argsort(-probsum)[0]

def train_on_pool(choice_function, X, y,  pool_idcs, train_idcs, test_idcs, name):
	Xtest, ytest = X[test_idcs], y[test_idcs]
	accuracies, balances, n_points, train_idcs, pool_idcs = list(), list(), list(), copy(train_idcs), copy(pool_idcs)

	gp = GaussianProcessClassifier(n_restarts_optimizer=25, kernel=Matern(),  n_jobs=-1, random_state=42)

	#Add initial points

	while pool_idcs:
		Xtrain, ytrain = X[train_idcs], y[train_idcs]
		gp.fit(Xtrain, ytrain)

		preds = gp.predict(Xtest)

		accuracies.append(accuracy_score(ytest, preds))
		n_points.append( len(train_idcs) )

		train_classes = np.unique(y[train_idcs], return_counts = True)[1]
		balances.append( max(train_classes)/sum(train_classes) )
		print(f"{len(train_idcs)}: {name}: {accuracies[-1]:.3}, class balance: {balances[-1]:.3}")


		y_pool_p = gp.predict_proba(X[pool_idcs])
		chosen_idx = choice_function(y_pool_p)

		train_idcs.append( pool_idcs.pop(chosen_idx) )

	return n_points, accuracies, balances


def run_experiment():
	np.random.seed(42)
	classes = 3
	X, y, pool_idcs, test_idcs, country_names, feature_names = dataprepare(bins=classes)

	print(f"pool size: {len(pool_idcs)}, test size: {len(test_idcs)}. Features: {feature_names}")


	methods = [random, least_conf, margin, entropy]
	names = ["Random", "Least Confident", "Margin", "Entropy"]

	#Start out with a country from each class
	train_idcs = []
	for c in range(classes):
		for i, p in enumerate(pool_idcs):
			if y[p] == c:
				print(f"Class {c} example: {country_names[p]} ")
				train_idcs.append(pool_idcs.pop(i))
				break
	balances, accuracies, n_points = list(), list(), list()

	for i, choice_method in enumerate(methods):
		print(f"Running {names[i]}")
		n_point, accuracy, balance = train_on_pool(choice_method, X, y, pool_idcs, train_idcs, test_idcs, names[i])
		accuracies.append(accuracy); n_points.append(n_point); balances.append(balance)


	for i, accuracy in enumerate(accuracies):
		plt.plot(n_points[i], accuracy)
	plt.legend(names)
	plt.title("Uncertainty sampling methods: Accuracy")

	plt.xlabel("Numer of Training Countries")
	plt.ylabel(f"Accuracy on {len(test_idcs)} test countries")
	plt.show()

	for i, balance in enumerate(balances):
		plt.plot(n_points[i], balance)
	plt.legend(names)
	plt.title("Uncertainty sampling methods: Class balancing")

	plt.axhline(y=0.33, xmin=n_point[0], xmax=n_point[-1], linestyle='--')
	plt.xlabel("Numer of Training Countries")
	plt.ylabel(f"Training set balance: Largest proportion")
	plt.show()

if __name__ == "__main__":
	run_experiment()



