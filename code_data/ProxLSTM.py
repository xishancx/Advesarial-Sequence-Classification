import torch
import torch.nn as nn
import torch.autograd as ag
from scipy import optimize
from scipy.optimize import check_grad
import numpy


class ProximalLSTMCell(ag.Function):
    def __init__(self, lstm: nn.LSTMCell):
        super(ProximalLSTMCell, self).__init__()
        self.lstm = lstm   # use LSTMCell as blackbox

    def forward(self, input, pre_h, pre_c):
        return self.lstm(input, (pre_h, pre_c))

    def backward(self, grad_h, grad_c):
        pass

