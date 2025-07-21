import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DATASETS_ROOT: str = '~/Repositories/Dataset'

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root=DATASETS_ROOT, train=True, transform=transforms, download=False)
test_dataset = datasets.MNIST(root=DATASETS_ROOT, train=False, transform=transforms, download=False)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

random_item = train_dataset.__getitem__(0)
print(f'Image shape: {random_item[0].shape}, Label: {random_item[1]}')
plt.imshow(random_item[0].squeeze(), cmap='gray')
plt.show()

class SimpeCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(input=1, output=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Shape: [batch_size, 32, 14, 14]
        
        self.conv2 = nn.Conv2d(input=32, output=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Shape: [batch_size, 64, 7, 7]
        
        # Fully connected layers
        self.fc1 = nn.Linear(input=64 * 7 * 7, output=128)
        self.fc2 = nn.Linear(input=128, output=10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x