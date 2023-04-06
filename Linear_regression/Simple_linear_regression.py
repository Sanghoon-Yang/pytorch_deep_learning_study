import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#훈련데이터셋
xtrain = torch.FloatTensor([[1],[2],[3]])
ytrain = torch.FloatTensor([[2],[4],[6]])

#가중치 편향 초기화
W = torch.zeros(1, requires_grad=True) #requires_grad = 변수가 학습을 통해 변화하는 변수임을 선언
b = torch.zeros(1, requires_grad=True)

#경사하강법 구현
optimizer = optim.SGD([W,b], lr=0.1) #학습대상인 W,b가 SGD 의 입력

nb_epochs = 2000

for epoch in range(nb_epochs+1):

    #hypothesis 정립
    hp = xtrain*W + b

    #cost function
    cost = torch.mean((ytrain-hp)**2) #평균제곱오차



    #gradient를 0으로 초기화
    optimizer.zero_grad()

    #cost function을 미분하여 gradient 계싼
    cost.backward()

    #W, b update
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))

