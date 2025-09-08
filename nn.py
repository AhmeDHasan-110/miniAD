import random
import math
import numpy as np
from rad_engine import Arrow

random.seed(12)

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = []
    
    def parameters(self):
        if self._parameters:
            for p in self._parameters['weights'] + [self._parameters['bias']] :
                yield p
        else:
            for m in self._modules:
                yield from m.parameters()
    
    def zero_grad(self):
        for p in self._parameters:
            p.adjoint = 0


class Neuron(Module):
    def __init__(self, n_weights, activation):
        super().__init__()
        self._parameters['weights'] = [Arrow(random.gauss(0, 1)) for i in range(n_weights)]
        self._parameters['bias'] = Arrow(0)
        self.activation = activation
    
    def __call__(self, x):
        out = 0
        for xi, wi in zip(x, self._parameters['weights']):
            out += wi * xi
        out += self._parameters['bias']

        return out.relu() if self.activation == 'relu' else out.tanh() if self.activation == 'tanh' else out

    def __repr__(self):
        n_inputs = len(self._parameters['weights'])
        return f"Neuron(inputs={n_inputs}, activation={self.activation}, bias={self._parameters['bias'].value:.4f})"



class Layer(Module):
    def __init__(self, n_inputs, n_neurons, activation):
        super().__init__()
        self._modules = [Neuron(n_inputs, activation) for _ in range(n_neurons)]
    
    def __call__(self, x):
        out = [node(x) for node in self._modules]
        return out
    
    def parameters(self):
        return list(super().parameters())
    
    def __repr__(self):
        return f"Layer(n_neurons={len(self._modules)}, activation={self._modules[0].activation if self._modules else None})"



class NeuralNetwork(Module):
    def __init__(self, input_dim, hidden_dim:list, activation='relu'):
        super().__init__()
        dims = zip([input_dim] + hidden_dim, hidden_dim)
        last = len(hidden_dim)-1
        self._modules = [Layer(i, o, activation=activation if idx != last else None) for idx, (i, o) in enumerate(dims)]

    def __call__(self,x):
        assert all(isinstance(i, (list, np.ndarray)) for i in x), "input must be of dimension 2"
        out = []
        for inp in x:
            x_input = inp
            for layer in self._modules:
                output = layer(x_input)
                x_input = output
            out.append(output)
        return out
    
    def parameters(self):
        return list(super().parameters())

    def __repr__(self):
        layers_repr = ", ".join([f"{len(layer._modules)}" for layer in self._modules])
        return f"NeuralNetwork(layers=[{layers_repr}], activations={[n.activation for l in self._modules for n in l._modules[:1]]})"


class MSELoss:
    def __call__(self, pred, target):
        assert isinstance(pred[0], (list, np.ndarray)), "output prediction must be of shape(n,-1)"
        loss = 0
        for p,t in zip(pred, target):
            for p_val, t_val in zip(p, t):
                loss += (p_val - t_val) ** 2
        return loss / (len(pred) * len(pred[0]))
    
    def __repr__(self):
        return "MSELoss()"


class GradientDescent(Module):
    def __init__(self, params:list, lr:float=0.001):
        super().__init__()
        self._parameters = params
        self.lr = lr

    def step(self):
        for p in self._parameters:
            p.value -= p.adjoint * self.lr

    def __repr__(self):
        return f"GradientDescent(lr={self.lr}, n_params={len(self._parameters)})"
