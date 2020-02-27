import numpy as np
from scipy.spatial.distance import cdist
from GPyOpt.models.gpmodel import GPModel
import matplotlib.pyplot as plt
def load_data():
	text_data = """ 
	3500  0.4423657706097197  Sigmoid  0.6415
	600  0.05357242679336405  ReLU  0.6135
	800  0.024286531237800668  Sigmoid  0.624
	900  0.5359999678433617  Tanh  0.621
	900  0.6064720199905345  Tanh  0.6215
	3500  0.28760052084452836  Sigmoid  0.639
	3500  0.4812338027466384  Sigmoid  0.649
	3500  0.598166666688632  Sigmoid  0.654
	3500  0.7073019913319729  Sigmoid  0.635
	3500  0.6366823572272423  Sigmoid  0.6455
	3500  0.7208836418636946  Sigmoid  0.6425
	3500  0.8135979227010048  Sigmoid  0.6045

	2900  0.2120309217292945  Sigmoid  0.6535
	1300  0.16575828954515115  ReLU  0.6465
	200  0.9318047460286568  Tanh  0.335
	2200  0.7223805260392725  ReLU6  0.4905
	1200  0.05956534512526279  ReLU6  0.6355000000000001
	2900  1.0  Sigmoid  0.135
	2900  0.18851940843012052  Sigmoid  0.6545
	1300  0.5317365663075784  Sigmoid  0.665
	1300  1.0  Sigmoid  0.099
	1300  0.37484134186025747  Sigmoid  0.6635
	1200  0.43302270846068275  ReLU6  0.6145
	1200  0.9814408341697629  ReLU6  0.097

	3300  0.20745359704991084  ReLU  0.619
	2900  0.48178037973857923  Sigmoid  0.6355
	1700  0.37725127529617486  ReLU6  0.5955
	3400  0.5452219321218295  Sigmoid  0.6275
	1800  0.47970862684081317  ReLU  0.613
	2900  0.4817801130255398  Sigmoid  0.634
	2900  0.4891192307830994  Sigmoid  0.642
	2900  0.4947118739255611  Sigmoid  0.6365
	2900  0.5834934845352279  Sigmoid  0.651
	2900  0.6168247073529851  Sigmoid  0.6365
	2900  1  Sigmoid  0.0965
	2900  0.5604663271240786  Sigmoid  0.635
	"""
	# activ_translate = {'Tanh': [1, 0, 0, 0],
	# 		'ReLU': [0, 1, 0, 0],
	# 		'ReLU6': [0, 0, 1, 0],
	# 		'Sigmoid': [0, 0, 0, 1],
	# 		}

	acq_texts = text_data.strip().split('\n\n')

	for i, acq_text in enumerate(acq_texts):
		acq_texts[i] = acq_text.strip().split('\n') 
		for j, datapoint in enumerate(acq_texts[i]):
			values = datapoint.strip().split()
						
			acq_texts[i][j] = [int(values[0]),  float(values[1]) ]
			#acq_texts[i][j] +=  activ_translate[values[2]]
			acq_texts[i][j].append( float(values[3]))
	data = np.array(acq_texts)
	return data


def squared_exponential_kernel(x, y, lengthscale, variance):
    # COMPUTE ALL THE PAIRWISE DISTANCES, size: NxM
    sqdist = cdist(x.reshape(-1,1), y.reshape(-1,1))
    # COMPUTE THE KENEL, this should also be NxM
    k = variance * np.exp(-sqdist**2 / (lengthscale**2))
    return k


if __name__ == "__main__":

	data = load_data()
	for acq_idx in range(data.shape[0]):
		data_i = data[acq_idx]
		X = data_i[:, :-1]
		Y = data_i[:, -1:]
	
		gp = GPModel(verbose=False)
		for i in range(1, X.shape[0]):
			gp.updateModel(X, Y, X[i], Y[i])
		
		
		

		N = 200
		plot_points = []

		x, y = np.linspace(0, 5000, N), np.linspace(0, 1, N)
		
		for x_ in x :
			for y_ in y:
				plot_points.append([x_,y_])

		plot_X = np.array(plot_points)
		
		xx, yy = np.meshgrid(x, y)

		plot_Z, _ = gp.predict(plot_X)
		plot_Z = plot_Z.reshape((N,N))

		plt.contourf(xx, yy, plot_Z)
		plt.show()