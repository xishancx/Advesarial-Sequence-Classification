
import torch
import torch.nn as nn
import torch.autograd as ag

from torch.autograd import Variable
from torch.nn import functional as F


class ProxLSTMCell(ag.Function):

    @staticmethod
    def forward(ctx, h_t, s_t, G_t, prox_epsilon=1):
        # s_t = s_t.unsqueeze(2)
        # G_t_transpose = torch.transpose(G_t, 1, 2)
        mul = torch.matmul(G_t, G_t.T)
        one_eye = torch.eye(mul.shape[-1])
        one_eye = one_eye.reshape((one_eye.shape[0], one_eye.shape[0]))
        # one_eye = one_eye.repeat(h_t.shape[0], 1, 1)
        inv = torch.inverse(one_eye + prox_epsilon*mul)
        # print(inv.shape, s_t.shape)
        c_t = (s_t.T @ inv).T
        # c_t = c_t.squeeze()
        # print(c_t.shape, one_eye.shape)
        ctx.save_for_backward(h_t, c_t, G_t, inv)

        return (h_t, c_t)


    @staticmethod
    def backward(ctx, grad_h, grad_c):
        h_t, c_t, G_t, inv = ctx.saved_tensors

        # print(inv.shape, grad_c.shape, grad_h.shape)

        a = (grad_c.T @ inv).T  # torch.matmul

        grad_g1 = torch.matmul(a, c_t.T)
        grad_g2 = torch.matmul(c_t, a.T)

        grad_g = -torch.matmul(grad_g1 + grad_g2, G_t)
        grad_s = (grad_c.T @ inv).T


        return grad_h, grad_s, grad_g, None