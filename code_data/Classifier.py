
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
# from ProxLSTM import ProximalLSTMCell
import ProxLSTM as pro
import torch.autograd as ag


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
		# self.prox_lstm = ProximalLSTMCell(self.lstm)
		self.linear = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, r, batch_size, epsilon=5, mode='plain'):
		# do the forward pass
		# pay attention to the order of input dimension.
		# input now is of dimension: batch_size * sequence_length * input_size

		if mode == 'plain':  # epsilon is 0
			# chain up the layers
			out = self.normalize(input)  # N x L x C
			out = out.permute((0, 2, 1))  # conv1d need N x C x L
			out = self.conv(out)
			out = self.relu(out)
			out = out.permute((2, 0, 1))  # lstm need L x N x C
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
			out = out.permute((0, 2, 1))  # N x C x L
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

		if mode == 'ProxLSTM':
			prox = pro.ProxLSTMCell.apply
			out = F.normalize(input)
			# # Dropout layer
			# if self.apply_dropout:
			#     normalized = self.dropout(normalized)
			out = self.conv(out.permute(0, 2, 1)).permute(2, 0, 1)

			with torch.enable_grad():
				self.v = self.relu(out).requires_grad_(True)
				# # Batch Norm layer
				# if self.apply_batch_norm:
				#     self.lstm_input = self.batch_norm(self.lstm_input.permute(0, 2, 1))
				#     self.lstm_input = self.lstm_input.permute(0, 2, 1)
				self.h_t = torch.zeros(self.v.shape[1], self.hidden_size)  # h_0
				self.c_t = torch.zeros(self.v.shape[1], self.hidden_size)  # c_0
				for v_t in self.v:
					self.h_t, self.s_t = self.lstm(v_t, (self.h_t, self.c_t))
					self.G_t = torch.zeros(v_t.shape[0], self.lstm.hidden_size, self.lstm.input_size)
					# for i in range(self.s_t.size(-1)):
					self.G_t = ag.grad(self.s_t, v_t, grad_outputs=torch.ones_like(self.s_t), create_graph=True, retain_graph=True)[0]
					self.h_t, self.c_t = prox(self.h_t, self.s_t, self.G_t, epsilon)
			out = self.linear(self.h_t)
			return out
