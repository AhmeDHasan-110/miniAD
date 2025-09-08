
# miniAD

This project is my **own implementation of a reverse-mode automatic differentiation engine**, built completely from scratch in Python.  
It is inspired by how frameworks like PyTorch work under the hood, but kept small and simple so that the core ideas are easy to see.  

The project has two main parts:

---

## 1. `rad_engine.py` — Reverse-Mode Autodiff Engine
- Defines the **`Arrow`** class, which works like a tiny version of PyTorch’s `Tensor`.  
- Supports building a **computation graph** during the forward pass.  
- Implements **reverse-mode backpropagation** with `.backward()`.  
- Currently supports the main operations:
  - Addition, subtraction, multiplication, division, power  
  - Activation functions: ReLU, tanh  
- Each operation stores a local “backward rule,” and gradients are propagated backwards through the graph.  

---

## 2. `nn.py` — Simple Neural Network Library
This file provides a very small neural network framework:

- **`Module`** → Base class for managing parameters.  
- **`Neuron`** → A single neuron with weights, bias, and activation.  
- **`Layer`** → A collection of neurons.  
- **`NeuralNetwork`** → A feed-forward neural network built from layers.  
- **`MSELoss`** → Mean squared error loss function.  
- **`GradientDescent`** → A simple optimizer to update weights using gradients.  

This lets you define and train small neural networks directly using the autodiff engine.

---

## ☑️ Tests
The **`tests`** folder is for the sake of verifications. It contains two files:  
- **`test_rad_engine.py`** → compares the autodiff engine on a very small network that uses different operations (add, multiply, ReLU, tanh, power, …) against the same network in PyTorch.  

- **`test_nn.py`** → compares the `NeuralNetwork` class on a slightly larger network against PyTorch, using both the engine and the nn module together.  

---

## Why I built this
I wanted to understand how deep learning libraries like PyTorch actually work behind the scenes. Writing my own **reverse-mode autodiff engine** and simple neural network framework gave me hands-on experience with:  
- Computation graphs  
- Backpropagation  
- Parameter management  
- Loss functions and optimizers  


This is a **minimal educational project**, not meant for production use.  
The autodiff engine is limited to a few core operations, but can be extended.  
