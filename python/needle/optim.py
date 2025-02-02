"""Optimization module"""
import needle
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue
            if param not in self.u:
                self.u[param] = 0
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * (param.grad.data + self.weight_decay * param.data)
            param.data = param.data - self.lr * self.u[param]
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            if param not in self.m:
                self.m[param] = 0
            if param not in self.v:
                self.v[param] = 0
            g = (param.grad + self.weight_decay * param).detach()
            self.m[param] = (self.beta1 * self.m[param] + (1 - self.beta1) * g).detach()
            self.v[param] = (self.beta2 * self.v[param] + (1 - self.beta2) * g**2).detach()
            m_hat = (self.m[param] / (1 - (self.beta1 ** self.t))).detach()
            v_hat = (self.v[param] / (1 - (self.beta2 ** self.t))).detach()
            param.data = param.data - self.lr * (m_hat / (v_hat**0.5 + self.eps))
        ### END YOUR SOLUTION
