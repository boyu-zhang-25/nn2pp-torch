from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
import matplotlib.pyplot as plt

from teacher_dataset import *
from dpp_sample_expected import *

# the student network
# switch between soft committee machine and two-layer FFNN
class student_MLP(nn.Module):

	def __init__(self,
				input_dim, 
				hidden_size,
				nonlinearity,
				mode,
				device):
		super(student_MLP, self).__init__()

		# MLP with one hidden layer
		self.w1 = nn.Linear(input_dim, hidden_size, bias = False) 
		self.w2 = nn.Linear(hidden_size, 1, bias = False) 
		self.input_dim = input_dim

		self.mode = mode
		self.hidden_size = hidden_size

		# choose between ['linear', 'sigmoid', and 'relu']
		if nonlinearity in ['linear', 'sigmoid', 'relu']:
			self.nonlinearity = nonlinearity
		else:
			print('Activation function not supported.')
			raise ValueError

		# nonlinearity layer
		if self.nonlinearity == 'relu':
			self.activation = nn.ReLU()
		elif self.nonlinearity == 'sigmoid':
			# use torch.ERF instead
			pass
		else:
			pass			

		self.device = device
		self.initialize()

	# initialization
	def initialize(self):

		if self.nonlinearity == 'sigmoid':
			nn.init.normal_(self.w1.weight.data)
		else:
			nn.init.normal_(self.w1.weight.data, std = 1 / math.sqrt(self.input_dim))

		if self.mode == 'soft_committee':
			print('soft committee machine (freeze student w2 as 1.0)')
			nn.init.ones_(self.w2.weight.data)
			self.w2.weight.requires_grad = False
		else:
			print('two-layer FFNN (DO NOT freeze student w2)')
			if self.nonlinearity == 'sigmoid':
				nn.init.normal_(self.w2.weight.data)
			else:
				nn.init.normal_(self.w2.weight.data, std = 1 / math.sqrt(self.input_dim))
				

	# forward call
	def forward(self, x):

		h = self.w1(x)

		# normalization and ERF activation following [S. Goldt, 2019]
		h_norm = h / math.sqrt(self.input_dim)
		h_norm_new = h_norm / math.sqrt(2)
		a = torch.erf(h_norm_new)
		output = self.w2(a)

		return output

# training loop
# online learning SGD
def train(args, model, device, train_loader, criterion, optimizer, epoch):

	model.train()

	current_error = 0

	# train_loader: input_dim * num_training_data
	for idx in range(len(train_loader)):

		# single data point
		data, target = train_loader[idx]
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		output = model(data)


		# print(output.shape, target.shape)
		loss = criterion(output, target.view(-1))
		loss.backward()

		# manually scale gradient for auto grad 
		# auto grad does not consider the square in MSE and the normalization in [S. Goldt, 2019]
		if model.w2.weight.requires_grad:

			model.w2.weight.grad = model.w2.weight.grad / 2
			model.w1.weight.grad = model.w1.weight.grad * math.sqrt(args.input_dim) / 2

		else:
			model.w1.weight.grad = model.w1.weight.grad * math.sqrt(args.input_dim) / 2

		optimizer.step()

		current_error += loss.item()

		if idx % 100000 == 0 and idx != 0:
			print('Train Example: [{}/{}]\tLoss: {:.6f}\t Epoch[{}]'.format(idx, len(train_loader), current_error, epoch))
			current_error = 0

# testing loop
def test(args, model, device, teacher_dataset, criterion):

	model.eval()
	current_error = 0

	test_num = teacher_dataset.test_inputs.shape[1]
	for idx in range(test_num):

		data, target = teacher_dataset.get_test_example(idx)
		data, target = data.to(device), target.to(device)
		output = model(data)

		loss = criterion(output, target.view(-1))
		current_error += loss.item()

	return current_error / test_num


def main():

	# hyperparameter settings
	parser = argparse.ArgumentParser(description='Teacher-Student Setup')

	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--student_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--nonlinearity', type = str, help='choice of the activation function')
	parser.add_argument('--mode', type = str, help='soft_committee or normal')

	# optimization setup
	parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
						help='learning rate (default: 0.0001)')
	parser.add_argument('--momentum', type=float, default = 0, metavar='M',
						help='SGD momentum (default: 0)')
	parser.add_argument('--epoch', type = int, default = 1, help='number of epochs')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	# data storage
	parser.add_argument('--trained_weights', type = str, default = 'place_holder', help='path to the trained weights for loading')
	parser.add_argument('--teacher_path', type = str, help='Path to the teacher network (dataset).')
	args = parser.parse_args()

	# print(args)
	# CUDA
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print('device', device)

	# reproducibility
	torch.manual_seed(args.seed)

	# create the model, loss, and optimizer
	model = student_MLP(input_dim = args.input_dim,
						hidden_size = args.student_h_size,
						nonlinearity = args.nonlinearity,
						mode = args.mode,
						device = device).to(device)

	# soft committee machine; freeze the second layer
	if args.mode == 'soft_committee':
		print('soft_committee!')
		model.w2.weight.requires_grad = False
		optimizer = optim.SGD([
								{'params': [model.w1.weight], 'lr' : args.lr / np.sqrt(args.input_dim)},
								], lr = args.lr, momentum = args.momentum)
	else:
		print('two-layers NN!')
		optimizer = optim.SGD([
								{'params': [model.w1.weight], 'lr' : args.lr / np.sqrt(args.input_dim)},
								{'params': [model.w2.weight], 'lr' : args.lr / args.input_dim}
								], lr = args.lr, momentum = args.momentum)
	criterion = nn.MSELoss()

	if torch.cuda.device_count() > 1 and args.procedure == 'training':
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		# model = nn.DataParallel(model)

	# get the data set from the teacher network
	# it is just the teacher dataset... bad naming :)
	train_loader = pickle.load(open(args.teacher_path, "rb" ))

	# train 
	if args.procedure == 'training':
		print('Training started!')

		# online SGD
		for epoch in range(args.epoch):
			train(args, model, device, train_loader, criterion, optimizer, epoch)

		torch.save(model.state_dict(), 'student_' + str(args.student_h_size) + '.pth')


	# testing
	else:
		print('Testing:', args.pruning_choice, '\nstudent hidden size:', args.student_h_size, '\nk:', args.k)

		# inference only
		with torch.no_grad():

			# load the unpruned model and masks
			file_name = 'student_masks_' + args.pruning_choice + '_' + str(args.student_h_size) + '.pkl'
			# TODO


if __name__ == '__main__':
	main()
