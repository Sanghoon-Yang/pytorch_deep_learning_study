import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
def sigmoid(x):
    sig = 1/(1+np.exp(-x))
    return sig

x = np.arange(-5,5,0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.show()
'''

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True) # 크기는 2 x 1
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W,b], lr=0.01)
nb_epochs = 3001

for epoch in range(nb_epochs):

    hp = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    #hypothesis = torch.sigmoid(x_train.matmul(W)+b)
    cost = F.binary_cross_entropy(hp, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))