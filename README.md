# nn2pp-torch

This is an UNOFFICIAL reimplementation of the paper 'Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup' [Goldt, et al., 2019] in Pytorch.

## Teacher-student setup

### Generating dataset and the teacher network:
>python3 teacher_dataset.py --input_dim 500 --teacher_h_size 5 --teacher_path teacher.pkl --num_data 6000 --mode normal

with the following arguments:
```
	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--teacher_h_size', type = int, help='hidden layer size of the student MLP.')
	parser.add_argument('--num_data', type = int, help='Number of data points to be genrated.')
	parser.add_argument('--mode', type = str, help='soft_committee or normal')
	parser.add_argument('--sig_w', type = float, help='scaling variable for the output noise.')

	# data storage
	parser.add_argument('--teacher_path', type = str, help='Path to store the teacher network (dataset).')
```

Please set `mode` to `soft_committee` for SCM or `normal` for two-layer FFNN.

The teacher default activation function is the ERF.


### Training the student network:
>python3 teacher_student.py --input_dim 500 --student_h_size 5 --teacher_path teacher.pkl  --nonlinearity sigmoid  --mode normal  --epoch 1 --lr 0.05

with the following arguments:
```
	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--student_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--nonlinearity', type = str, help='choice of the activation function')
	parser.add_argument('--mode', type = str, help='soft_committee or normal')

	# optimization setup
	parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
						help='learning rate (default: 0.0001)')
	parser.add_argument('--momentum', type=float, default = 0, metavar='M',
						help='SGD momentum (default: 0)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	# data storage
	parser.add_argument('--teacher_path', type = str, help='Path to the teacher network (dataset).')
```
The student only supports ERF activation for now.

# Copyright

* [1] S. Goldt, M.S. Advani, A.M. Saxe, F. Krzakala, L. Zdeborová, NeurIPS 2019
  (forthcoming), [arXiv:1906.08632](https://arxiv.org/abs/1906.08632)


