from copy import deepcopy
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, supervised_labels=False):
        self.supervised_labels = supervised_labels
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.precision_matrices = self.precision_init()

        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def precision_init(self):
        omega_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            omega_matrices[n] = variable(p.data)
        return omega_matrices

    def precision_update(self, model, criterion, args, data_loader):
        print('\nUpdating Omega precision matrix for EWC...')
        model.eval()
        for batch_images, batch_labels, _ in data_loader:
            model.zero_grad()
            outputs = model(batch_images.cuda(), args.trn_n)
            if self.supervised_labels:
                labels = batch_labels
            else:
                labels = outputs.max(1)[1]  # use unsupervised labels
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad: self.precision_matrices[n] += p.grad.data ** 2

        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def penalty(self, model: nn.Module):
        loss = variable(torch.zeros(1))
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self.precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
