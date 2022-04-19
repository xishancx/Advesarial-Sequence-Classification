
import torch
import torch.nn as nn
from torch.nn import functional as F
import ProxLSTM as pro
import torch.autograd as ag


class LSTMClassifier(nn.Module):

	def __init__(
		self
		, output_size: int
		, hidden_size: int
		, input_size: int
	):
		"""

		:param output_size: the number of classes, determines the size of the output tensor
		:param hidden_size: the number of features in the hidden state of the LSTM
		:param input_size: the input size of the
		"""

		super(LSTMClassifier, self).__init__()

		self.input_size = input_size
		self.output_size = output_size  # should be 9
		self.hidden_size = hidden_size  # the dimension of the LSTM output layer should be 12
		self.normalize = F.normalize
		self.dropout = nn.Dropout(p=0.2)
		self.b_norm = nn.BatchNorm1d(64)
		self.conv = nn.Conv1d(
			in_channels=self.input_size,
			out_channels=64,
			kernel_size=3,
			stride=1
		)
		self.relu = nn.ReLU()
		self.lstm = nn.LSTMCell(64, hidden_size)
		self.linear = nn.Linear(self.hidden_size, self.output_size)

	def forward(
			self
			, input: torch.Tensor
			, r: torch.Tensor
			, batch_size: int
			, epsilon: float = 1
			, mode: str = 'plain'
			) -> torch.Tensor:
		"""

		:param input: input batch of sequences
		:param r: perturbation applied for adversarial training
		:param batch_size: batch size
		:param epsilon: epsilon hyperparameter for Adversarial and Proximal modes
		:param mode: "plain" for vanilla LSTM, "AdvLSTM" for adversarial training or "ProxLSTM" for proximal mapping
		:return: output tensor of size batch_size
		"""

		# do the forward pass
		# pay attention to the order of input dimension.
		# input now is of dimension: batch_size * sequence_length * input_size

		if mode == 'plain':  # epsilon is 0
			# chain up the layers
			out = self.normalize(input)  # N x L x C
			out = out.permute((0, 2, 1))  # conv1d need N x C x L
			out = self.conv(out)
			# print("Plain out size", out.size())
			out = self.b_norm(out)
			out = self.relu(out)
			out = out.permute((2, 0, 1))  # lstm need L x N x C
			h_t, c_t = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
			for i in range(out.size(0)):  # loop through L
				h_t, c_t = self.lstm(out[i], (h_t, c_t))
			out = self.linear(h_t)
			return out

		if mode == 'AdvLSTM':  # epsilon is applied to perturbation
			# chain up the layers
			# different from mode='plain', you need to add r to the forward pass
			# also make sure that the chain allows computing the gradient with respect to the input of LSTM
			out = self.normalize(input)  # N x L x C.
			out = out.permute((0, 2, 1))  # N x C x L
			out = self.conv(out)
			out = self.b_norm(out)
			out = self.relu(out)
			self.v = torch.tensor(out.permute((2, 0, 1)), requires_grad=True)  # save the input to the lstm layer
			out = self.v + epsilon * r  # perturb the input to lstm layer
			h_t, c_t = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
			for i in range(out.size(0)):  # loop through L
				h_t, c_t = self.lstm(out[i], (h_t, c_t))
			out = self.linear(h_t)
			return out

		if mode == 'ProxLSTM':
			prox = pro.ProxLSTMCell.apply
			out = F.normalize(input).permute((0, 2, 1))
			out = self.dropout(out)
			out = self.conv(out)
			out = self.b_norm(out)
			out = self.relu(out).permute((2, 0, 1))
			out = self.dropout(out)

			with torch.enable_grad():
				self.v = out.requires_grad_(True)
				h_t = torch.zeros(self.v.shape[1], self.hidden_size)  # h_0
				s_t = torch.zeros(self.v.shape[1], self.hidden_size)  # s_0

				for v_t in self.v:
					h_t, s_t = self.lstm(v_t, (h_t, s_t))
					G_t = ag.grad(s_t, v_t, grad_outputs=torch.ones_like(s_t), create_graph=True, retain_graph=True)[0]
					h_t, c_t = prox(h_t, s_t, G_t, epsilon)

			out = self.dropout(h_t)
			out = self.linear(out)

			return out
