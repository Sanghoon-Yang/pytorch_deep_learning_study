import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    device = 'cuda'


# mnist_train = dsets.MNIST(root='MNIST_data/',
#                           train=True,
#                           download=True)
#
# mnist_test = dsets.MNIST(root='MNIST_data/',
#                          train=False,
#                          download=True)
#

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
mnist.target = mnist.target.astype(np.int8)

X = mnist.data / 255  # 0-255값을 [0,1] 구간으로 정규화
y = mnist.target

X_train, X_test, Y_train,Y_test = train_test_split(X, y, test_size=1/7, random_state=1)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(Y_train)
y_test = torch.LongTensor(Y_test)

train_data = TensorDataset(X_train,y_train)
test_data = TensorDataset(X_test, y_test)

loader_train = DataLoader(train_data, batch_size=64, shuffle=True)
loader_test = DataLoader(test_data, batch_size=64, shuffle=False)


model = nn.Sequential()
model.add_module('fc1', nn.Linear(784,100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100,10))

n_epochs = 2000
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for data, targets in loader_train:

        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()

    print("epoch{}：완료\n".format(epoch))


def test():
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in loader_test:
            out = model(data)
            _, predicted = torch.max(out.data, 1)

            correct += predicted.eq(targets.data.view_as(predicted)).sum()

    data_num = len(loader_test.dataset)

    print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct,
                                                         data_num, 100. * correct / data_num))

test()

for epoch in range(3):
    train(epoch)

test()