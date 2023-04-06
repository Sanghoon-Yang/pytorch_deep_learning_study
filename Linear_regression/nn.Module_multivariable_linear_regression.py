import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr=0.00001)

nb_epochs = 5000

for epoch in range(nb_epochs+1):
    hp = model(x_train)
    cost = F.mse_loss(y_train,hp)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


new_var= torch.FloatTensor([[73, 80, 75]])

pred = model(new_var)
print(pred)