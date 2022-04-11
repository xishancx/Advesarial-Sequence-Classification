
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import ProxLSTM as pro


class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, input_size):
		super(LSTMClassifier, self).__init__()

		self.input_size = input_size
		self.output_size = output_size  # should be 9
		self.hidden_size = hidden_size  # the dimension of the LSTM output layer should be 12
		self.normalize = F.normalize
		self.conv = nn.Conv1d(
			in_channels=self.input_size,
			out_channels=64,
			kernel_size=3,
			stride=1
		)
		self.relu = nn.ReLU()
		self.lstm = nn.LSTMCell(64, hidden_size)
		pro.ProximalLSTMCell.reset(self.lstm)
		self.linear = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, r, batch_size, epsilon=0, mode='plain'):
		# do the forward pass
		# pay attention to the order of input dimension.
		# input now is of dimension: batch_size * sequence_length * input_size

		if mode == 'plain':  # epsilon is 0
			# chain up the layers
			out = self.normalize(input)  # N x L x C
			out = torch.permute(out, (0, 2, 1))  # conv1d need N x C x L
			out = self.conv(out)
			out = self.relu(out)
			out = torch.permute(out, (2, 0, 1))  # lstm need L x N x C
			hx, cx = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
			for i in range(out.size(0)):  # loop through L
				hx, cx = self.lstm(out[i], (hx, cx))
			out = hx  # last time step
			out = self.linear(out)
			return out

		if mode == 'AdvLSTM':  # epsilon is applied to perturbation
			# chain up the layers
			# different from mode='plain', you need to add r to the forward pass
			# also make sure that the chain allows computing the gradient with respect to the input of LSTM
			out = self.normalize(input)  # N x L x C
			out = torch.permute(out, (0, 2, 1))  # N x C x L
			out = self.conv(out)
			out = self.relu(out)
			self.v = torch.tensor(torch.permute(out, (2, 0, 1)), requires_grad=True)  # save the input to the lstm layer
			out = self.v + epsilon * r  # perturb the input to lstm layer
			hx, cx = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
			for i in range(out.size(0)):  # loop through L
				hx, cx = self.lstm(out[i], (hx, cx))
			out = hx  # last time step
			out = self.linear(out)
			return out

		if mode == 'ProxLSTM':  # epsilon is lambda^(-1) * delta^2
			# chain up layers, but use ProximalLSTMCell here
			out = self.normalize(input)  # N x L x C
			out = torch.permute(out, (0, 2, 1))  # conv1d need N x C x L
			out = self.conv(out)
			out = self.relu(out)
			out = torch.permute(out, (2, 0, 1))  # prox lstm need L x N x C
			hx, cx = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
			for i in range(out.size(0)):  # loop through L
				hx, cx = pro.ProximalLSTMCell.apply(out[i], hx, cx, epsilon)
			out = hx  # last time step
			out = self.linear(out)
			return out
