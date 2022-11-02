"""The module.
"""
from typing import List, Callable, Any

import needle.ops
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weight features with kaiming uniform
        self.weight = Parameter(init.kaiming_uniform(fan_in=self.in_features, fan_out=self.out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(ops.transpose(init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype)))

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mul = ops.matmul(X, self.weight)
        if hasattr(self, "bias"):
            return mul + self.bias.broadcast_to(mul.shape)
        return mul
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return needle.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return needle.ops.ReLU()(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        result = x
        for module in self.modules:
            result = module(result)
        return result
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(logits.shape[-1], y, device=logits.device, dtype=logits.dtype)
        part1 = needle.ops.logsumexp(logits, axes=(-1,)) # (5, 1)
        part2 = needle.ops.summation(logits * one_hot, (-1,)) # (5, 10)
        return  needle.ops.summation(part1 - part2) / part1.shape[0]

        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x => B X d
        if self.training:
            mean = needle.summation(x, (0,)) / x.shape[0] # d
            numerator = x - mean.broadcast_to(x.shape)
            var = needle.summation(needle.power_scalar(numerator, 2), (0,)) / x.shape[0]
            batch_norm = numerator / needle.broadcast_to(needle.power_scalar(var + self.eps, 0.5), numerator.shape)
            self.running_mean = self.momentum * mean.detach() + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var.detach() + (1 - self.momentum) * self.running_var

            return self.weight.broadcast_to(batch_norm.shape) * batch_norm + self.bias.broadcast_to(batch_norm.shape)
        else:
            numerator = x - self.running_mean.data.broadcast_to(x.shape)
            batch_norm = numerator / needle.broadcast_to(needle.power_scalar(self.running_var.data + self.eps, 0.5), numerator.shape)
            return self.weight.broadcast_to(batch_norm.shape) * batch_norm + self.bias.broadcast_to(batch_norm.shape)

        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype).reshape((1, dim)))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype).reshape((1, dim)))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x -> B X d
        # mean -> B
        # (B X d) - (B,)
        mean = needle.ops.summation(x, (-1, )).broadcast_to(tuple(reversed(x.shape))).transpose() / x.shape[1] # B X d
        numerator = (x - mean) # B X d
        var = needle.ops.summation(needle.ops.power_scalar(numerator, 2), (-1, )) / x.shape[1] # B
        var = var.broadcast_to(tuple(reversed(numerator.shape))).transpose() # B X d
        ln = (numerator / needle.ops.power_scalar(var + self.eps, 0.5))
        return self.weight.broadcast_to((x.shape[0], self.dim)) * ln + self.bias.broadcast_to((x.shape[0], self.dim))
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            y = x * 1.0/(1-self.p)
            return y * init.randb(*x.shape, p=1-self.p)
        else:
            return x

        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION



