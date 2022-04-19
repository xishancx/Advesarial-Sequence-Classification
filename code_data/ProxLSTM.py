from typing import Tuple

import torch
import torch.autograd as ag


class ProxLSTMCell(ag.Function):

    @staticmethod
    def forward(
        ctx
        , h_t: torch.Tensor
        , s_t: torch.Tensor
        , G_t: torch.Tensor
        , prox_epsilon: float = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param ctx: the context of autograd function
        :param h_t: LSTM hidden state of previous time step
        :param s_t: LSTM output of previous time step
        :param G_t: gradient of hidden state with respect to the input
        :param prox_epsilon: epsilon hyperparameter for proximal mapping
        :return: a tuple of hidden state and cell output after proximal mapping
        """

        mul = torch.matmul(G_t, G_t.T)
        one_eye = torch.eye(mul.shape[-1])
        one_eye = one_eye.reshape((one_eye.shape[0], one_eye.shape[0]))
        inv = torch.inverse(one_eye + prox_epsilon*mul)
        c_t = (s_t.T @ inv).T
        ctx.save_for_backward(h_t, c_t, G_t, inv)

        return h_t, c_t

    @staticmethod
    def backward(
        ctx
        , grad_h: torch.Tensor
        , grad_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param ctx: the context of autograd function
        :param grad_h: gradient of the hidden state of time step t+1
        :param grad_c: gradient of the output of the cell of time step t+1
        :return: a tuple consisting of gradient of hidden state, gradient of LSTM cell output and hessian of LSTM cell output
        """

        h_t, c_t, G_t, inv = ctx.saved_tensors
        a = (grad_c.T @ inv).T

        grad_g1 = torch.matmul(a, c_t.T)
        grad_g2 = torch.matmul(c_t, a.T)

        grad_g = -torch.matmul(grad_g1 + grad_g2, G_t)
        grad_s = (grad_c.T @ inv).T

        return grad_h, grad_s, grad_g
