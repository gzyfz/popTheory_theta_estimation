import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
class SNPMatrixDataset(Dataset):
    def __init__(self, matrix_dir, max_height, max_width):
        self.matrix_files = [os.path.join(matrix_dir, fname) for fname in os.listdir(matrix_dir) if fname.endswith('.npy')]
        self.max_height = max_height
        self.max_width = max_width
    
    def __len__(self):
        return len(self.matrix_files)
    
    def __getitem__(self, idx):
        matrix_path = self.matrix_files[idx]
        snp_matrix = np.load(matrix_path)
        
        # Padding
        padded_matrix = np.zeros((self.max_height, self.max_width), dtype=np.float32)
        padded_matrix[:snp_matrix.shape[0], :snp_matrix.shape[1]] = snp_matrix
        
        theta = float(matrix_path.split('theta')[1].split('.npy')[0])
        
        # Adding a channel dimension and ensuring correct shape
        return torch.tensor(padded_matrix, dtype=torch.float32).unsqueeze(0), torch.tensor(theta, dtype=torch.float32)

class ComplexCNN(nn.Module):
    def __init__(self, max_height, max_width):
        super(ComplexCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(int(max_height/4)*int(max_width/4)*64, 1000)
        self.fc2 = nn.Linear(1000, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Set the maximum dimensions based on your dataset analysis
max_height = 300  # Example value, set to your dataset's max height
max_width = 300  # Example value, set to your dataset's max width

# Initialize the dataset and data loader
matrix_dir = '../raw_data/snp_matrices'
snp_dataset = SNPMatrixDataset(matrix_dir, max_height, max_width)
snp_loader = DataLoader(snp_dataset, batch_size=4, shuffle=True)

# Instantiate the model
model = ComplexCNN(max_height, max_width)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
loss_values = []
num_epochs = 8

for epoch in range(num_epochs):
    for i, (matrices, thetas) in enumerate(snp_loader):
        optimizer.zero_grad()
        outputs = model(matrices)
        loss = criterion(outputs.squeeze(), thetas)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(snp_loader)}], Loss: {loss.item()}')
        loss_values.append(loss.item())

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Loss Function Over Time')
plt.legend()
plt.show()

input = None

for i, (matrices, thetas) in enumerate(snp_loader):
    if i == 0:
        input = matrices

from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter instance (logs will be saved in the 'runs' directory)
writer = SummaryWriter('runs/model_visualization')


# To use add_graph, you need to provide a dummy input tensor that matches the input shape the model expects
# For example, if your model expects a 3-channel image of size 224x224
dummy_input = input

# Add the model graph to TensorBoard
writer.add_graph(model, dummy_input)

# Close the writer
writer.close()