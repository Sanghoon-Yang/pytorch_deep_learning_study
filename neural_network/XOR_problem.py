import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


tf = torch.cuda.is_available()
if tf:
    torch.cuda.manual_seed_all(1)
    device = 'cuda'
else:
    device = 'cpu'


X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

model_single_perceptron = nn.Sequential(
    nn.Linear(2,1, bias=True),
    nn.Sigmoid()
).to(device)

model_multi_layer_perceptron = nn.Sequential(nn.Linear(2,50, bias=True),
                                             nn.Sigmoid(),
                                             nn.Linear(50,30,bias=True),
                                             nn.Sigmoid(),
                                             nn.Linear(30,1,bias=True),
                                             nn.Sigmoid()).to(device)



criterion = nn.BCELoss().to(device)

optimizer_slp = optim.SGD(model_single_perceptron.parameters(), lr=1)
optimizer_mlp = optim.SGD(model_multi_layer_perceptron.parameters(), lr=1)

for epoch in range(2000):
    hp = model_single_perceptron(X)
    cost = criterion(hp, Y)


    optimizer_slp.zero_grad()
    cost.backward()
    optimizer_slp.step()

    if epoch % 100 == 0: # 100번째 에포크마다 비용 출력
        print(epoch, cost.item())

print('-'*100)

for epoch in range(2000):
    hp = model_multi_layer_perceptron(X)
    cost = criterion(hp, Y)


    optimizer_mlp.zero_grad()
    cost.backward()
    optimizer_mlp.step()

    if epoch % 100 == 0: # 100번째 에포크마다 비용 출력
        print(epoch, cost.item())


with torch.no_grad():
    hp = model_single_perceptron(X)
    #print(hp)
    predicted = (hp > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hp.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())

    print('-'*100)
    hp = model_multi_layer_perceptron(X)
    #print(hp)
    predicted = (hp > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hp.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())


