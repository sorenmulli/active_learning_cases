from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import make_friedman2
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern
import numpy as np

N = 10000
addns = 2
X, _ = make_friedman2(n_samples=N, noise=0, random_state=0)
Xtrain = X[:int(0.7*N)]
Xpool = X[int(0.7*N):int(0.8*N)]
Xtest = X[int(0.8*N):]
Xtrain = (Xtrain - np.mean(Xtrain)) / np.std(Xtrain)
Xtest = (Xtest - np.mean(Xtest)) / np.std(Xtest)

y = np.random.randint(2, size = N)
ytrain = y[:int(0.7*N)]
ypool = X[int(0.7*N):int(0.8*N)]
ytest = y[int(0.7*N):]

model = GaussianProcessClassifier()
gp = model.fit(Xtrain, ytrain)
preds = gp.predict_proba(Xtrain)


preds = np.max(preds, axis = 1)
newXs = (1 - preds).argsort()[-addns:][::-1]