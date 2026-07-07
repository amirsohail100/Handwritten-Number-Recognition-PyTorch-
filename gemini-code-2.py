import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. DATA LOAD & TRANSFORM (Custom Dataset Class banana)
# PyTorch mein load aur transform aksar ek hi class ke andar handle kiya jata hai.
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        # Data ko NumPy se PyTorch Tensors mein convert karna (Loading)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        # Yeh batata hai ki dataset mein total kitne samples hain
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # --- DATA TRANSFORMATION ---
        # Yahan aap apna transformation logic likhte hain (e.g., scaling, augmentation)
        feature = feature * 2.0  # Har feature ko 2 se multiply kiya
        
        return feature, label

# Dummy NumPy data create karna (1000 samples, 10 features)
np_features = np.random.rand(1000, 10).astype(np.float32)
np_labels = np.random.randint(0, 2, size=(1000, 1)).astype(np.float32)

# Dataset ka object banana
my_dataset = CustomDataset(np_features, np_labels)


# 2. BATCHING & PREFETCHING (DataLoader ka use)
# DataLoader aapke data ko automatically shuffle, batch aur parallelly load karta hai.
BATCH_SIZE = 32
data_loader = DataLoader(
    my_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,          # Data ko har epoch par shuffle karega
    num_workers=2,         # Multi-processing (Parallel data loading ke liye CPU cores)
    pin_memory=True        # GPU training ko fast karne ke liye memory optimize karta hai
)


# 3. DATA FEED (Model banakar usme data feed karna)
# Ek simple Neural Network definition
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.sigmoid(x)

# Model, Loss, aur Optimizer initialize karna
model = SimpleModel()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop (Data Feed Stage)
print("--- Model Training Started ---")
EPOCHS = 5

for epoch in range(EPOCHS):
    running_loss = 0.0
    
    # DataLoader se batches ko loop mein nikalna aur model ko feed karna
    for batch_features, batch_labels in data_loader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass (Data Feed)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass aur optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(data_loader):.4f}")