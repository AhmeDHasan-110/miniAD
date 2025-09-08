import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rad_engine import Arrow
from nn import NeuralNetwork, MSELoss, GradientDescent

LR = 0.01
BATCHSIZE = 3

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def my_nn():
    net = NeuralNetwork(input_dim=4, hidden_dim=[3, 3, 2], activation="relu")

    # call torch_net function before the weights of my network change
    pytorch_net ,pytorch_loss = torch_nn(net)

    criterion = MSELoss()
    optimizer = GradientDescent(net.parameters(), lr=LR)

    losses = []
    for epoch in range(1):  # 1 epoch for demo

        for i in range(0, len(X), BATCHSIZE):
            xb = X[i:i+BATCHSIZE]
            yb = Y[i:i+BATCHSIZE]

            preds = net(xb)
            loss = criterion(preds, yb)
            losses.append(loss.value)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
    
    return net, losses, pytorch_net ,pytorch_loss


def torch_nn(ref_net):
    net = Net()
    
    # extract my network weights
    weights, biases = [], []
    for layer in ref_net._modules:
        cur_weights, cur_bias = [], []
        for n in layer._modules:
            cur_weights.append(list(map(lambda x: x.value , n._parameters['weights'])))
            cur_bias.append(n._parameters['bias'].value)
        weights.append(cur_weights)
        biases.append(cur_bias)

    # Set the weights
    with torch.no_grad():
        net.fc1.weight[:] = torch.tensor(weights[0])
        net.fc1.bias[:] = torch.tensor(biases[0])
        net.fc2.weight[:] = torch.tensor(weights[1])
        net.fc2.bias[:] = torch.tensor(biases[1])
        net.fc3.weight[:] = torch.tensor(weights[2])
        net.fc3.bias[:] = torch.tensor(biases[2])

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=LR)

    losses = []
    for epoch in range(1):

        for i in range(0, len(X), BATCHSIZE):
            xb = X[i:i+BATCHSIZE]
            yb = Y[i:i+BATCHSIZE]

            preds = net(torch.from_numpy(xb))
            loss = criterion(preds, torch.from_numpy(yb))
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return net, losses


if __name__ == '__main__':

    X = np.array([
    [1.0, 2.0, 3.0, 4.6],
    [7.0, 3.8, 4.0, 5.0],
    [3.0, 4.0, 12.0, 1.0],
    [4.2, 9.0, 6.0, 7.0],
    [5.5, 6.0, 7.0, 8.0],
    [6.0, 11.0, 2.0, 9.0]
    ])

    # Targets: 6 examples, each with 2 outputs
    Y = np.array([
        [4.0, 0.0],
        [1.0, 3.0],
        [5.7, 1.0],
        [0.0, 1.0],
        [1.0, 1.2],
        [3.5, 1.0]
    ])

    my_net ,my_loss, pytorch_net, pytorch_loss = my_nn()

    my_weights = []
    for i, p in enumerate(my_net.parameters()[:15]):
        if not (i+1) % 5 : # ignore bias
            continue
        my_weights.append(p.value)

    pytorch_weights = pytorch_net.fc1.weight.detach().numpy().reshape(1, -1)[0]

    # compare the losses of the 2 batches
    assert np.allclose(my_loss, pytorch_loss, atol= 1e-5), "losses don't match !"

    # compare the weights of the first layer at each network
    assert np.allclose(my_weights, pytorch_weights, atol= 1e-5), "weights don't match !"

    print("all works")