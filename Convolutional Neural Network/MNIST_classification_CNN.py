import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(1)
else:
    device = 'cpu'


learning_rate = 0.001
training_epochs = 15
batch_size = 100

class myCNNmodel(nn.Module):
    def __init__(self):
        super(myCNNmodel, self). __init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1,32,kernel_size = 3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32,64, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*64,10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


model = myCNNmodel().to(device)

mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.CrossEntropyLoss().to(device)
total_batch = len(data_loader)


for epoch in range(training_epochs):
    avg_loss = 0
    for x,y in data_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        out = criterion(pred, y)
        out.backward()
        optimizer.step()
        avg_loss += out / total_batch
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_loss))


with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())