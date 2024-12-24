import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import idx2numpy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


device = torch.device("cude" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.layer1 = nn.Linear(784, 1024)
        self.layer2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x
    

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.layer1 = nn.Linear(1024, 512)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(256,128)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(128, 64)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.dropout(x)
        return x
    
data_dir = 'mnist_data/MNIST/raw' 

def load_mnist():
    train_images = idx2numpy.convert_from_file(f'{data_dir}/train-images-idx3-ubyte')
    train_images = train_images.astype(np.float32) / 255

    train_labels = idx2numpy.convert_from_file(f'{data_dir}/train-labels-idx1-ubyte')
    train_labels = train_labels.astype(np.float32)

    test_images = idx2numpy.convert_from_file(f'{data_dir}/t10k-images-idx3-ubyte')
    test_images = test_images.astype(np.float32)/255

    test_labels = idx2numpy.convert_from_file(f'{data_dir}/t10k-labels-idx1-ubyte')
    test_labels = test_labels.astype(np.float32)

    train_images = torch.tensor(train_images).view(-1, 784)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_images = torch.tensor(test_images).view(-1, 784)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True)

    return train_loader, test_loader


train_loader, test_loader = load_mnist()

client_model = ClientModel().to(device)
server_model = ServerModel().to(device)

loss = nn.CrossEntropyLoss()
client_opt = optim.Adam(client_model.parameters(), lr = 0.001)
server_opt = optim.Adam(server_model.parameters(), lr = 0.001)


for epoch in range(50):
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

        loss_val.backward()
        server_opt.step()

        intermediate_grad = intermediate.grad.clone()

        intermediate.backward(intermediate_grad)
        client_opt.step()
    print(f'Epoch {epoch+1}, Loss: {loss_val.item()}')


client_model.eval()
server_model.eval()
correct = 0; total = 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.view(data.size(0), -1)
        intermediate_output = client_model(data)
        output = server_model(intermediate_output)

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy: {100 * correct / total}%')
print(correct)
print(total)