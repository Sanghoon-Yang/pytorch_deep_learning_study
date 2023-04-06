import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1,1)

optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 3000

for epoch in range(nb_epochs+1):

    prediction = model(x_train)
    cost = F.mse_loss(prediction,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


new_var = torch.FloatTensor([[4]])

pred = model(new_var)
print(pred)
print(list(model.parameters()))