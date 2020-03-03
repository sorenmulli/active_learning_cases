import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor


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
	for acq_idx in range(data.shape[0]):
		data_i = data[acq_idx]
		X = data_i[:, :-1]
		Y = data_i[:, -1:]
		
		X[:, 0] /= 5000
		gp = GaussianProcessRegressor(random_state=42).fit(X, Y)
		
		N = 200
		plot_points = []

		x, y = np.linspace(0, 5000, N), np.linspace(0, 1, N)
		
		for x_ in x :
			for y_ in y:
				plot_points.append([x_/5000,y_])

		plot_X = np.array(plot_points)
		xx, yy = np.meshgrid(x, y)
		plot_Z  = gp.predict(plot_X)
		plot_Z = plot_Z.reshape((N,N))


		fig = plt.figure()
		zmin,zmax = 0, 1

		cs = plt.contourf(xx, yy, -plot_Z,  vmin=zmin,vmax=zmax)
		plt.scatter(X[:,0]*5000, X[:, 1], cmap=cs.cmap, vmin=zmin,vmax=zmax)

		#plt.colorbar(cvmin=zmin,vmax=zmax)
		plt.show()