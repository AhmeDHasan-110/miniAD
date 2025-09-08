import torch
import torch.nn as nn
import numpy as np
from rad_engine import Arrow

def test_engine():
    x, y = Arrow(1), Arrow(2)

    n1 = x**2 + y.relu()
    n2 = x*1.5 - y.tanh()

    n3 = n1*1 + n2*3
    n4 = n1*5 + n2*4

    e = (n3*1) ** (n4*2)
    e.backward()

    arrow_grads = [x.adjoint, y.adjoint]

    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)

    n1 = x**2 + y.relu()
    n2 = x*1.5 - y.tanh()

    n3 = n1*1 + n2*3
    n4 = n1*5 + n2*4

    e = (n3*1) ** (n4*2)
    e.backward()

    torch_grads = [x.grad, y.grad]

    assert np.allclose(arrow_grads, torch_grads, atol= 1e-5), "Gradients are not equal"
    print("All works")

if __name__ == '__main__':
    test_engine()