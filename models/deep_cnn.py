import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
# Transformation pipeline to resize SNP matrices and convert them to tensors
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel to make it 3-channel
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom Dataset class for SNP matrices
class SNPMatrixDataset(Dataset):
    def __init__(self, matrix_dir, transform=None):
        self.matrix_files = [os.path.join(matrix_dir, fname) for fname in os.listdir(matrix_dir) if fname.endswith('.npy')]
        self.transform = transform
    
    def __len__(self):
        return len(self.matrix_files)
    
    def __getitem__(self, idx):
        # Load the SNP matrix
        matrix_path = self.matrix_files[idx]
        snp_matrix = np.load(matrix_path)
        
        # Normalize and prepare the matrix
        snp_matrix = snp_matrix.astype(np.float32)
        snp_matrix = torch.tensor(snp_matrix)  # Convert to tensor
        snp_matrix = torch.unsqueeze(snp_matrix, 0)  # Add a channel dimension
        
        if self.transform:
            snp_matrix = self.transform(snp_matrix)  # Apply transformations
        
        theta = float(matrix_path.split('theta')[1].split('.npy')[0])
        return snp_matrix, torch.tensor(theta, dtype=torch.float32)


# Define the complex CNN model
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Adjust for 3 input channels
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)  # Example size adjustment for 224x224 input after pooling
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 28 * 28)  # Adjust the flattening based on your pooling and input size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize dataset and DataLoader
matrix_dir = '../raw_data/snp_matrices'  # Adjust directory path
snp_dataset = SNPMatrixDataset(matrix_dir, transform=transform)
snp_loader = DataLoader(snp_dataset, batch_size=4, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = ComplexCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
loss_values = []
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(snp_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        loss_values.append(loss.item())
        print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(snp_loader)}, Loss: {loss.item()}')

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Loss Function Over Time')
plt.legend()
plt.show()
