import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import idx2numpy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.fc1 == nn.Linear(784,128)
    
    def forward(self, x):
        x = self.fc1(x)
        return x
    

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.fc2(x)
        return x
    
data_dir = 'mnist_data/MNIST/raw' 

def load_mnist():
    train_images = idx2numpy.convert_from_file(f'{data_dir}/train-images-idx3-ubyte')
    train_images = train_images.astype(np.float32) / 255

    train_labels = idx2numpy.convert_from_file(f'{data_dir}/train-labels-idx1-ubyte')
    train_labels = train_labels.astype(np.float32)/255

    test_images = idx2numpy.convert_from_file(f'{data_dir}/t10k-images-idx3-ubyte')
    test_images = test_images.astype(np.float32)/255

    test_labels = idx2numpy.convert_from_file(f'{data_dir}/t10k-labels-idx1-ubyte')
    test_labels = test_labels.astype(np.float32)/255

    train_images = torch.tensor(train_images).view(-1, 784)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_images = torch.tensor(test_images).view(-1, 784)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

    return train_loader, test_loader


train_loader, test_loader = load_mnist()

client_model = ClientModel()
server_model = ServerModel()

loss = nn.CrossEntropyLoss()
client_opt = optim.Adam(client_model.parameters(), lr = 0.001)
server_opt = optim.Adam(server_model.parameters(), lr = 0.001)


for epoch in range(5):
    client_model.train()
    server_model.train()

    for data, target in train_loader:
        client_opt.zero_grad()
        server_opt.zero_grad()

        data = data.view(data.size(0), -1)
        intermediate = client_model(data)
        intermediate = intermediate.detach().requires_grad_()

        output = server_model(intermediate)
        loss_val = loss(output, target)

        loss.backward()
        server_opt.step()

        intermediate_grad = intermediate.grad.clone()

        intermediate.backward(intermediate_grad)
        client_opt.step()
    print(f'Epoch {epoch+1}, Loss: {loss_val.item()}')



