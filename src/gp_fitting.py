import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def load_data():
	text_data = """ 
	  3917  0.6923  1  0.5165  
	  1805  0.1127  0  0.609  
	  1217  0.7398  2  0.5765  
	  2618  0.6605  1  0.5605  
	  2639  0.491  0  0.6115  
	  2220  0.3325  0  0.6185  
	  1002  0.4978  0  0.5965  
	  3354  0.1637  0  0.609  
	  1  0.1262  1  0.1145  
	  2863  0.5179  0  0.5595  
	  2540  0.1569  0  0.6275000000000001  
	  3251  1.0  2  0.0835  

	 2403  0.6196  3  0.647  
	 130  0.7724  1  0.202  
	 2706  0.4818  0  0.5685  
	 554  0.03571  2  0.6245  
	 2983  0.5369  1  0.6145  
	 2397  0.6166  3  0.6275  
	 2407  0.6219  3  0.6385  
	 2471  0.6537  3  0.6435  
	 2601  0.7138  3  0.636  
	 3794  0.4693  1  0.611  
	 2641  0.6416  3  0.6285  
	 2176  0.69  3  0.6355  

	 2835  0.316  2  0.616  
	 3046  0.01182  3  0.627  
	 934  0.6308  0  0.6005  
	 1068  0.5681  3  0.641  
	 848  0.5295  1  0.6175  
	 1150  0.5441  3  0.646  
	 1297  0.5009  3  0.635  
	 1165  0.5443  3  0.6455  
	 1089  0.5296  3  0.646  
	 1115  0.5386  3  0.6425  
	 1127  0.5375  3  0.649  
	 1112  0.5333  3  0.636  
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
			acq_texts[i][j].append( -float(values[3]))
	data = np.array(acq_texts)
	return data


if __name__ == "__main__":

	data = load_data()
	acqs_str = ['EI', 'MPI', 'LCB']
	for acq_idx in range(data.shape[0]):
		data_i = data[acq_idx]
		X = data_i[:, :-1]
		Y = data_i[:, -1:]
		
		X[:, 0] /= 5000
		gp = GaussianProcessRegressor(random_state=42, normalize_y=True, n_restarts_optimizer=100, kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2)).fit(X, Y)
		
		N = 200
		plot_points = []

		x, y = np.linspace(0, 5000, N), np.linspace(0, 1, N)
		for x_ in x :
			for y_ in y:
				plot_points.append([x_/5000,y_])

		plot_X = np.array(plot_points)
		xx, yy = np.meshgrid(x, y)
		Z  = gp.predict(plot_X)
		plot_Z = Z.reshape((N,N))

		plt.rcParams.update({'font.size': 22})
		fig, ax = plt.subplots()
		contour = ax.contourf(xx, yy, -plot_Z, np.arange(0.5, 0.75, .001), extend = 'both')

		fig.colorbar(contour, ax=ax, label = 'GP Point estimate of accuracy')
		
		ax.plot(X[:,0]*5000, X[:, 1], 'r*')
		for j in range(X.shape[0]):
			ax.annotate(f'{j+1}: {-Y[j][0]*100:.3} %', (X[j,0]*5000, X[j,1]))


		ax.set_title(f'GP fitted to accuracies acquired by {acqs_str[acq_idx]}',y=1.08)
		ax.set_xlabel('N hidden neurons')
		ax.set_ylabel('Dropout probability')
		
		plt.show()