from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
import matplotlib.pyplot as plt

from teacher_student import *
from teacher_dataset import *


def get_Q(path_to_student, path_to_teacher, input_dim):

	
	model.load_state_dict(torch.load(args.trained_weights, map_location = torch.device('cpu')))
	w1 = stu.w1.weight.data.cpu().numpy() # hid_dim * inp_dim
	hid_dim, inp_dim = w1.shape[0], w1.shape[1]
	print('student w1 size:', w1.shape, 'mask size:', mask_list[0].T.shape)
	Q = np.dot(w1, w1.T) / input_dim

	# get the teacher net
	teacher = pickle.load(open(path_to_teacher, 'rb'))
	teahcer_w1 = teacher.w1.data.cpu().numpy() # teacher_hid_dim * input_dim
	print('teacher w1 size:', teahcer_w1.shape)
	
	T = np.dot(teahcer_w1, teahcer_w1.T) / input_dim
	return Q, T

def plot_Q(expected_Q, unpurned_Q, teacher_Q):

	plt.figure(1)
	fig, ax = plt.subplots()
	im = ax.imshow(expected_Q)

	# Loop over data dimensions and create text annotations.
	for i in range(len(expected_Q)):
		for j in range(len(expected_Q)):
			text = ax.text(j, i, '%.3f'%expected_Q[i, j],
						   ha="center", va="center", color="w")

	ax.set_title("expected_Q")
	fig.tight_layout()
	plt.savefig('expected_Q.png')

	plt.figure(2)
	fig, ax = plt.subplots()
	im = ax.imshow(unpurned_Q)

	# Loop over data dimensions and create text annotations.
	for i in range(len(unpurned_Q)):
		for j in range(len(unpurned_Q)):
			text = ax.text(j, i, '%.3f'%unpurned_Q[i, j],
						   ha="center", va="center", color="w")

	ax.set_title("unpurned_Q")
	fig.tight_layout()
	plt.savefig('unpurned_Q.png')

	plt.figure(3)
	fig, ax = plt.subplots()
	im = ax.imshow(teacher_Q)

	# Loop over data dimensions and create text annotations.
	for i in range(len(teacher_Q)):
		for j in range(len(teacher_Q)):
			text = ax.text(j, i, '%.3f'%teacher_Q[i, j],
						   ha="center", va="center", color="w")

	ax.set_title("teacher_Q")
	fig.tight_layout()
	plt.savefig('teacher_Q.png')

# 
def get_R(path_to_student_mask, path_to_teacher, input_dim):

	# get the student net
	unpurned_MLP, mask_list = pickle.load(open(path_to_student_mask, 'rb'))
	mask_num = len(mask_list)
	student_w1 = unpurned_MLP.w1.weight.data.cpu().numpy() # student_hid_dim * inp_dim
	student_hid_dim, inp_dim = student_w1.shape[0], student_w1.shape[1]
	print('student w1 size:', student_w1.shape, 'mask size:', mask_list[0].T.shape)

	# get the teacher net
	teacher = pickle.load(open(path_to_teacher, 'rb'))
	teahcer_w1 = teacher.w1.data.cpu().numpy().T # input_dim * teacher_hid_dim
	teacher_hid_dim = teahcer_w1.shape[1]
	print('teacher w1 size:', teahcer_w1.shape)

	'''
	teahcer_w1 = np.load('/Users/mac/Desktop/pyscm/scm_erf_erf_N500_M2_K5_lr0.5_wd0_sigma0_bs1_i1steps800_s0_teacher.npy')
	print('teacher loaded')
	teahcer_w1 = teahcer_w1.T
	'''

	# get the expected R on purned student_w1
	# student_hid_dim * teacher_hid_dim
	expected_R = np.zeros((student_hid_dim, teacher_hid_dim))
	for mask in mask_list:
		expected_R += np.dot(student_w1 * mask.T, teahcer_w1)
	expected_R = expected_R / mask_num

	# get the expected R on unpurned student_w1
	unpurned_R = np.dot(student_w1, teahcer_w1)

	# pickle.dump((expected_R, unpurned_R), open('expected_R', "wb"))
	return expected_R / input_dim, unpurned_R / input_dim

def plot_R(expected_R, unpurned_R):

	plt.figure(1)
	fig, ax = plt.subplots()
	im = ax.imshow(expected_R)

	# Loop over data dimensions and create text annotations.
	for i in range(len(expected_R)):
		for j in range(len(expected_R[1])):
			text = ax.text(j, i, '%.3f'%expected_R[i, j],
						   ha="center", va="center", color="w")

	ax.set_title("expected_R")
	fig.tight_layout()
	plt.savefig('expected_R.png')

	plt.figure(2)
	fig, ax = plt.subplots()
	im = ax.imshow(unpurned_R)

	# Loop over data dimensions and create text annotations.
	for i in range(len(unpurned_R)):
		for j in range(len(unpurned_R[1])):
			text = ax.text(j, i, '%.3f'%unpurned_R[i, j],
						   ha="center", va="center", color="w")

	ax.set_title("unpurned_R")
	fig.tight_layout()
	plt.savefig('unpurned_R.png')


def main():

	parser = argparse.ArgumentParser(description='Order Parameter')
	parser.add_argument('--path_to_student', type = str)
	parser.add_argument('--path_to_teacher', type = str, default = 'place_holder')
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	args = parser.parse_args()

	Q, T = get_Q(args.path_to_student, args.path_to_teacher, args.input_dim)
	plot_Q(Q, T)

	R = get_R(args.path_to_student, args.path_to_teacher, args.input_dim)
	plot_R(R)

if __name__ == '__main__':
	main()
