import sklearn.datasets as ds
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import TensorDataset, DataLoader, random_split
import argparse


class Lin(torch.nn.Module):
    def __init__(self, input):
        super(Lin, self).__init__()
        self.linear = nn.Linear(input, 2)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_epochs', type=int, default=30)
    parser.add_argument('-d', '--dev', type=str, default='cpu')
    return parser


def Validate(model, val_data):
    val_loader = DataLoader(dataset=val_data, batch_size=10, shuffle=True)
    model = model.eval()
    correct = 0
    total = 0
    for X_batch, y_batch in val_loader:
        y_pred = model(X_batch.float())
        total += y_batch.size(0)
        correct += (y_pred.argmax(1) == y_batch).sum().item()
    return correct / total


def train_model(model, train_data, val_data, max_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=True)
    for epoch in range(max_epochs):
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            model = model.train()
            y_pred = model(X_batch.float())
            loss1 = loss(y_pred, y_batch)
            total += y_batch.size(0)
            correct += (y_pred.argmax(1) == y_batch).sum().item()
            loss1.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 3 == 2:
            acc = correct / total
            print(f'Epoch = {epoch}, acc = {acc}, loss = {loss1}')
            print(f'VAL:  Epoch = {epoch},acc ={Validate(model,val_data)} ')


def main():
    parser = createParser()
    args = parser.parse_args()
    X, y = ds.load_digits(n_class=2, return_X_y=True)
    l = len(X)
    if args.dev == 'gpu':
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    X = torch.LongTensor(X)
    X.to(device)
    y = torch.LongTensor(y)
    y.to(device)
    data = TensorDataset(X, y)
    train_data, test_data, val_data = random_split(data, [int(l * 0.7), int(l * 0.2), l - int(l * 0.7) - int(l * 0.2)])
    input = 8 * 8
    model = Lin(input).to(device)
    train_model(model, train_data, val_data, args.max_epochs)


if __name__ == '__main__':
    main()
