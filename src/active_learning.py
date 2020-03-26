import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score

def dataprepare(bins=4, split=0.8):
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
	pool_idcs, test_idcs = scrampled_idcs[:cutoff], scrampled_idcs[cutoff:]

	return X, y.flatten(), pool_idcs, test_idcs, country_names, feature_names

	
random = lambda: np.random.permutation(range(len(ypool)))[0] 
least_conf = lambda probs: np.argsort(1 - probs.max(1))[0]
margin = lambda probs: np.argsort(probs.max(1)-probs.min(1))[-1]
entropy = lambda probs: probsum = probs[:,0]*np.log(probs[:,0])

#probsum = np.sum(ypool_p*np.log(ypool_p),axis = 1)
probsum = np.sum(np.prod(probs*np.log(probs),axis = 1),axis = 1)
probsum = ypool_p[:,0]*np.log(ypool_p[:,0])
for i in range(1,max(y)):
	probsum += ypool_p[:,i]*np.log(ypool_p[:,i])

def entropy(probs):
	probsum = probs[:,0]*np.log(probs[:,0])
	for i in range(1,max(y)):
		probsum += probs[:,i]*np.log(probs[:,i])
	return np.argsort(-probsum)[0]

def train_on_pool(choice_function, X, y,  pool_idcs, test_idcs, ini_train = 5):
	Xtest, ytest = X[test_idcs], y[test_idcs]
	accuracies, n_points, train_idcs = list(), list(), list()

	gp = GaussianProcessClassifier()

	#Add initial points
	train_idcs.extend( [pool_idcs.pop() for _ in range(ini_train)] )	
	
	while pool_idcs:
		Xtrain, ytrain = X[train_idcs], y[train_idcs]
		gp.fit(Xtrain, ytrain)
		
		preds = model.predict(Xtest)
		accuracies.append( sklearn.metrics.accuracy_score(ytest, preds) )
		
		y_pool_p = model.predict_proba(X_pool[pool_idcs])
		chosen_point = choice_function(y_pool_p)
		

	
def run_experiment():
	X, y, pool_idcs, test_idcs, country_names, feature_names = dataprepare()
	print(f"pool size: {len(pool_idcs)}, test size: {len(test_idcs)}. Features: {feature_names}")

	train_on_pool(choice_function, X, y, pool_idcs, test_idcs)

if __name__ == "__main__":
	run_experiment()



