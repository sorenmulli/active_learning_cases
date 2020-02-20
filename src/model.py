import torch
from torchvision import models, transforms, datasets
import time 
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
import os, sys
import copy

def setup_datatset():

	data_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

	batch_size = 4

	print("Initializing Datasets and Dataloaders...")


	cifar10 = datasets.CIFAR10('local_data/', train = True,  transform = data_transform, download = True)
	cifar10test = datasets.CIFAR10('local_data/', train = False,  transform = data_transform, download = True)

	train_size = int(0.8 * len(cifar10))
	test_size = len(cifar10) - train_size
	cifar10train, cifar10val = torch.utils.data.random_split(cifar10, [train_size, test_size])
	image_datasets = {'train': cifar10train, 'val': cifar10val}
	
	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
	test_loader = torch.utils.data.DataLoader(cifar10test, batch_size=batch_size, shuffle=True, num_workers=4)
	return dataloaders_dict, test_loader







def train_model(hyperparameters, model, dataloaders, num_epochs=25):
	in_features, out_features = 25088, 10
	
	p = hyperparameters['p']
	hidden_units = hyperparameters['hidden_units']
	activation_func = hyperparameters['activation_func']

	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	
	model.classifier[0] = torch.nn.Linear(in_features, hidden_units)
	model.classifier[1] = activation_func
	model.classifier[2] = torch.nn.Dropout(p=p)
	model.classifier[3] = torch.nn.Linear(hidden_units, hidden_units)
	model.classifier[4] = activation_func
	model.classifier[5] = torch.nn.Dropout(p=p)
	model.classifier[6] = torch.nn.Linear(hidden_units, out_features)

	print(model.classifier)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())

	val_acc_history = []

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		model.train()  # Set model to training mode

		running_loss = 0.0
		running_corrects = 0
		# Iterate over data.
		for inputs, labels in dataloaders['train']:
			inputs = inputs.to(device)
			labels = labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			with torch.set_grad_enabled(True):
				# Get model outputs and calculate loss
				outputs = model(inputs)
				loss = criterion(outputs, labels)


				loss.backward()
				optimizer.step()

			
			running_loss += loss.item() * inputs.size(0)

		epoch_loss = running_loss / len(dataloaders['train'].dataset)
		

		print('{epoch} Loss: {:.4f} '.format(epoch, epoch_loss))
		print()
		
	model.eval()
	corrects = 0
	for inputs, labels in dataloaders['val']:
		inputs = inputs.to(device)
		labels = labels.to(device)
		with torch.set_grad_enabled(False):
			# Get model outputs and calculate loss
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			corrects += torch.sum(preds == labels.data)

	val_acc = corrects.double() / len(dataloaders['test'].dataset)

	return model, val_acc

def test_finished_model(model, test_loader):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model.eval()
		
	corrects = 0
	for inputs, labels in test_loader:
		inputs = inputs.to(device)
		labels = labels.to(device)
		with torch.set_grad_enabled(False):
			# Get model outputs and calculate loss
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			corrects += torch.sum(preds == labels.data)

	test_acc = corrects.double() / len(dataloaders['test'].dataset)

	return test_acc 


if __name__ == "__main__":
	os.chdir(sys.path[0])

	model = models.vgg16(pretrained="JADAK!")
	for param in model.features.parameters(): param.requires_grad = False

	dataloaders, test_loader = setup_datatset()
	hyperparameters = {
		'activation_func' : torch.nn.ReLU(),
		'p': .5,
		'hidden_units': 10,
	}

	trained_model, val_acc  = train_model(hyperparameters, model, dataloaders)

