import torch
import torch.nn as nn


class ProximalLSTMCell(nn.Module):

    def __init__(self, lstm):
        super(ProximalLSTMCell, self).__init__()
        self.lstm = lstm
        self.grad_list = []

    # def reset(lstm: nn.LSTMCell):
    #     ProximalLSTMCell.lstm = lstm
    #     ProximalLSTMCell.grad_list = []

    def forward(self, x, pre_h, pre_c, epsilon):
        x = torch.tensor(x.clone().detach(), requires_grad=True)
        pre_h = torch.tensor(pre_h.clone().detach(), requires_grad=True)
        pre_c = torch.tensor(pre_c.clone().detach(), requires_grad=True)

        _, s = self.lstm(x, (pre_h, pre_c))
        print(s)
        s.backward()
        grad_Wh = [p.grad.clone().detach() for p in self.lstm.parameters()]
        ds_dpre_h = pre_h.grad.clone().detach()
        ds_dpre_c = pre_c.grad.clone().detach()
        G = x.grad.clone().detach()

        print(G.size(), s.size())
        post_c = torch.inverse(torch.ones(s.size()) + epsilon * torch.matmul(G, G.T)) * s
        print(post_c.size())

        post_h, _ = self.lstm(x, (pre_h, pre_c))
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

        self.grad_list.append((grad_Wh, ds_dpre_h, ds_dpre_c,
                                           grad_Wc, dh_dpre_h, dh_dpre_c,
                                           grad_s, dL_dG_dpre_c, dL_dG_dpre_h))

        return post_h, post_c

    def backward(self, grad_h, grad_c):
        grad_Wh, ds_dpre_h, ds_dpre_c, \
        grad_Wc, dh_dpre_h, dh_dpre_c, \
        grad_s, dL_dG_dpre_c, dL_dG_dpre_h = self.grad_list.pop(-1)

        for i, p in enumerate(self.lstm.parameters()):
            p.grad += (grad_h * grad_Wh[i] + grad_c * grad_Wc[i])

        grad_pre_h = grad_h * dh_dpre_h + dL_dG_dpre_h + grad_s * ds_dpre_h
        grad_pre_c = grad_h * dh_dpre_c + dL_dG_dpre_c + grad_s * ds_dpre_c

        return grad_pre_h, grad_pre_c

