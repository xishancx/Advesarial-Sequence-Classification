import torch
import torch.nn as nn
import torch.autograd as ag
from scipy import optimize
from scipy.optimize import check_grad
import numpy


class ProximalLSTMCell(ag.Function):
    lstm = None
    grad_list = None

    @staticmethod
    def reset(lstm: nn.LSTMCell):
        ProximalLSTMCell.lstm = lstm
        ProximalLSTMCell.grad_list = []

    @staticmethod
    def forward(ctx, x, pre_h, pre_c, epsilon):
        x = torch.tensor(x.clone().detach(), requires_grad=True)
        pre_h = torch.tensor(pre_h.clone().detach(), requires_grad=True)
        pre_c = torch.tensor(pre_c.clone().detach(), requires_grad=True)

        _, s = ProximalLSTMCell.lstm(x, (pre_h, pre_c))
        print(s)
        s.backward()
        grad_Wh = [p.grad.clone().detach() for p in ProximalLSTMCell.lstm.parameters()]
        ds_dpre_h = pre_h.grad.clone().detach()
        ds_dpre_c = pre_c.grad.clone().detach()
        G = x.grad.clone().detach()

        print(G.size(), s.size())
        post_c = torch.inverse(torch.ones(s.size()) + ProximalLSTMCell.epsilon * torch.matmul(G, G.T)) * s
        print(post_c.size())

        post_h, _ = ProximalLSTMCell.lstm(x, (pre_h, pre_c))
        post_h.backward()
        grad_Wc = [p.grad.clone().detach() for p in ProximalLSTMCell.lstm.parameters()]
        dh_dpre_h = pre_h.grad.clone().detach()
        dh_dpre_c = pre_c.grad.clone().detach()

        # this part is missing a term because I am not sure what gradient f(c) is
        a = torch.inverse(torch.ones(s.size()) + epsilon * torch.matmul(G, G.T))
        grad_s = a
        grad_G = -(a * post_c.T + post_c * a.T) * G

        Frobenius_product = grad_G * G
        Frobenius_product.backward()
        dL_dG_dpre_c = pre_c.grad.clone().detach()
        dL_dG_dpre_h = pre_h.grad.clone().detach()

        ProximalLSTMCell.grad_list.append((grad_Wh, ds_dpre_h, ds_dpre_c,
                                           grad_Wc, dh_dpre_h, dh_dpre_c,
                                           grad_s, dL_dG_dpre_c, dL_dG_dpre_h))

        return post_h, post_c

    @staticmethod
    def backward(ctx, grad_h, grad_c):
        grad_Wh, ds_dpre_h, ds_dpre_c, \
        grad_Wc, dh_dpre_h, dh_dpre_c, \
        grad_s, dL_dG_dpre_c, dL_dG_dpre_h = ProximalLSTMCell.grad_list.pop(-1)

        for i, p in enumerate(ProximalLSTMCell.lstm.parameters()):
            p.grad += (grad_h * grad_Wh[i] + grad_c * grad_Wc[i])

        grad_pre_h = grad_h * dh_dpre_h + dL_dG_dpre_h + grad_s * ds_dpre_h
        grad_pre_c = grad_h * dh_dpre_c + dL_dG_dpre_c + grad_s * ds_dpre_c

        return grad_pre_h, grad_pre_c

