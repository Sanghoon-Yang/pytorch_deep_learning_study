import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(device)

digits = load_digits()

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:5]): # 5개의 샘플만 출력
    plt.subplot(1, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i' % label)
plt.show()

X = digits.data # 이미지. 즉, 특성 행렬
Y = digits.target # 각 이미지에 대한 레이블

model = nn.Sequential(nn.Linear(64,32, bias=True),
                      nn.ReLU(),
                      nn.Linear(32,16, bias=True),
                      nn.ReLU(),
                      nn.Linear(16,10, bias=True),
                      nn.ReLU()).to(device)

X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.int64).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss = []
n_epochs = 500

for epoch in range(n_epochs+1):
    hp = model(X)
    cost = loss_function(hp, Y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, n_epochs, cost.item()
        ))

    loss.append(cost.item())

plt.plot(loss)
plt.show()