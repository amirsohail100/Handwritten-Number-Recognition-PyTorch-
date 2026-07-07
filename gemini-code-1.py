import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. LOAD & TRANSFORM (CSV ke liye Custom Dataset)
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        # Pandas se CSV load kiya
        df = pd.read_csv(csv_file)
        
        # Features (X) aur Labels (y) ko alag kiya
        # Maan lete hain aakhri column ka naam 'target' hai
        X = df.drop(columns=['target']).values
        y = df['target'].values
        
        # --- DATA TRANSFORMATION (Scaling/Normalization) ---
        # CSV data ke liye scaling bohot zaroori hoti hai
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Tensors mein convert kiya
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Shape (N, 1) banane ke liye

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# CSV Dataset ka object banaya (Apni file ka path yahan dein)
# dataset = CSVDataset("data.csv") 

# Dummy execution ke liye hum pehle ek fake CSV bana lete hain taaki code run ho sake
pd.DataFrame(dict(**{f'f{i}': [0.5]*100 for i in range(10)}, target=[1]*50 + [0]*50)).to_csv("data.csv", index=False)
dataset = CSVDataset("data.csv")

# 2. BATCHING (DataLoader)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)


# 3. MODEL DEFINE & FEED
class CSVModel(nn.Module):
    def __init__(self, input_dim):
        super(CSVModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 10 features hain hamare paas
model = CSVModel(input_dim=10)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop (Feeding)
print("--- PyTorch CSV Training Started ---")
for epoch in range(3):
    for batch_features, batch_labels in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Complete - Loss: {loss.item():.4f}")