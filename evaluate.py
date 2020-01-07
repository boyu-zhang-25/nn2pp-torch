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


def get_Q(path_to_mask_list, path_to_teacher, input_dim):

	unpruned_MLP, mask_list = pickle.load(open(path_to_mask_list, 'rb'))

	print('student w2:', unpruned_MLP.w2.weight.data)
	mask_num = len(mask_list)
	w1 = unpruned_MLP.w1.weight.data.cpu().numpy() # hid_dim * inp_dim
	hid_dim, inp_dim = w1.shape[0], w1.shape[1]
	print('student w1 size:', w1.shape, 'mask size:', mask_list[0].T.shape)

	# get the expected Q
	expected_Q = np.zeros((hid_dim, hid_dim))
	for mask in mask_list:
		purned_w = w1 * mask.T
		expected_Q += np.dot(purned_w, purned_w.T)
	expected_Q = expected_Q / mask_num
	expected_Q = expected_Q / input_dim

	# get the unpruned Q
	unpruned_Q = np.dot(w1, w1.T) / input_dim

	# get the teacher net
	teacher = pickle.load(open(path_to_teacher, 'rb'))
	teahcer_w1 = teacher.w1.data.cpu().numpy() # teacher_hid_dim * input_dim
	print('teacher w1 size:', teahcer_w1.shape)
	
	teacher_Q = np.dot(teahcer_w1, teahcer_w1.T) / input_dim



	return expected_Q, unpruned_Q, teacher_Q

def plot_Q(expected_Q, unpruned_Q, teacher_Q):

	plt.figure(1)
	fig, ax = plt.subplots()
	expected_Q = abs(expected_Q)
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
	unpruned_Q = abs(unpruned_Q)
	im = ax.imshow(unpruned_Q)

	# Loop over data dimensions and create text annotations.
	for i in range(len(unpruned_Q)):
		for j in range(len(unpruned_Q)):
			text = ax.text(j, i, '%.3f'%unpruned_Q[i, j],
						   ha="center", va="center", color="w")

	ax.set_title("unpruned_Q")
	fig.tight_layout()
	plt.savefig('unpruned_Q.png')

	plt.figure(3)
	fig, ax = plt.subplots()
	teacher_Q = abs(teacher_Q)
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
	unpruned_MLP, mask_list = pickle.load(open(path_to_student_mask, 'rb'))
	mask_num = len(mask_list)
	student_w1 = unpruned_MLP.w1.weight.data.cpu().numpy() # student_hid_dim * inp_dim
	student_hid_dim, inp_dim = student_w1.shape[0], student_w1.shape[1]
	print('student w1 size:', student_w1.shape, 'mask size:', mask_list[0].T.shape)

	# get the teacher net
	teacher = pickle.load(open(path_to_teacher, 'rb'))
	teahcer_w1 = teacher.w1.data.cpu().numpy().T # input_dim * teacher_hid_dim
	teacher_hid_dim = teahcer_w1.shape[1]
	print('teacher w1 size:', teahcer_w1.shape)
	

	# get the expected R on purned student_w1
	# student_hid_dim * teacher_hid_dim
	expected_R = np.zeros((student_hid_dim, teacher_hid_dim))
	for mask in mask_list:
		expected_R += np.dot(student_w1 * mask.T, teahcer_w1)
	expected_R = expected_R / mask_num

	# get the expected R on unpruned student_w1
	unpruned_R = np.dot(student_w1, teahcer_w1)

	# pickle.dump((expected_R, unpruned_R), open('expected_R', "wb"))
	return expected_R / input_dim, unpruned_R / input_dim

def plot_R(expected_R, unpruned_R):

	plt.figure(1)
	fig, ax = plt.subplots()
	expected_R = abs(expected_R)
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
	unpruned_R = abs(unpruned_R)
	im = ax.imshow(unpruned_R)

	# Loop over data dimensions and create text annotations.
	for i in range(len(unpruned_R)):
		for j in range(len(unpruned_R[1])):
			text = ax.text(j, i, '%.3f'%unpruned_R[i, j],
						   ha="center", va="center", color="w")

	ax.set_title("unpruned_R")
	fig.tight_layout()
	plt.savefig('unpruned_R.png')


def main():

	parser = argparse.ArgumentParser(description='Order Parameter')
	parser.add_argument('--path_to_student_mask', type = str)
	parser.add_argument('--path_to_teacher', type = str, default = 'place_holder')
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	args = parser.parse_args()

	expected_Q, unpruned_Q, teacher_Q = get_Q(args.path_to_student_mask, args.path_to_teacher, args.input_dim)	
	expected_R, unpruned_R = get_R(args.path_to_student_mask, args.path_to_teacher, args.input_dim)
	
	# Permute the matrix to make it block diagonal
	student_hid_dim, teacher_hid_dim = unpruned_R.shape
	z = int(student_hid_dim/teacher_hid_dim)
	unpruned_R_dash, unpruned_Q_dash, expected_R_dash ,expected_Q_dash = np.zeros((student_hid_dim,teacher_hid_dim)),  np.zeros((student_hid_dim,student_hid_dim)), np.zeros((student_hid_dim,teacher_hid_dim)),  np.zeros((student_hid_dim,student_hid_dim))
	dic = [[] for x in range(teacher_hid_dim)]
	for i in range(teacher_hid_dim):
		for j in range(student_hid_dim):
			if abs(unpruned_R[j][i])>=0.7:
				dic[i].append(j)

	print(dic,"hello")
	for x in range(teacher_hid_dim):
		for y in range(len(dic[x])):
			new_row = x*z+y
			cur = dic[x][y]
			print(new_row,cur)
			unpruned_R_dash[new_row,:] = unpruned_R[cur,:]
			expected_R_dash[new_row,:] = expected_R[cur,:]

	for x in range(student_hid_dim):
		for y in range(x+1):

			i = dic[int(x/z)][x%z]
			j = dic[int(y/z)][y%z]

			if x==y:
				unpruned_Q_dash[x][x] =  unpruned_Q[i][i]
				expected_Q_dash[x][x] =  expected_Q[i][i]
			else:
				unpruned_Q_dash[x][y] = unpruned_Q[i][j]
				unpruned_Q_dash[y][x] = unpruned_Q[i][j]

				expected_Q_dash[x][y] = expected_Q[i][j]
				expected_Q_dash[y][x] = expected_Q[i][j]

			# unpruned_Q[[new_row,cur],:] = unpruned_Q[[cur,new_row],:]			
			# expected_Q[[new_row,cur],:] = expected_Q[[cur,new_row],:]

			# unpruned_Q[:,[new_row,cur]] = unpruned_Q[:,[cur,new_row]]
			# expected_Q[:,[new_row,cur]] = expected_Q[:,[cur,new_row]]
	
	plot_Q(expected_Q_dash, unpruned_Q_dash, teacher_Q)
	plot_R(expected_R_dash, unpruned_R_dash)


if __name__ == '__main__':
	main()
