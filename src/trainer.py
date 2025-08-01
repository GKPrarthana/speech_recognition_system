import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import create_model
import torch.optim as optim
import torch.nn as nn

X = np.load('../data/processed/X.npy')  
y = np.load('../data/processed/y.npy') 

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

input_size = 13 
hidden_size = 128 
num_classes = 10 
learning_rate = 0.001
epochs = 20

model = create_model(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train() 
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    model.eval() 
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad(): 
        for batch in val_loader:
            X_batch, y_batch = batch
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)
    
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {total_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {100 * correct / total:.2f}%, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Accuracy: {100 * val_correct / val_total:.2f}%")

torch.save(model.state_dict(), '../models/speech_recognition_model.pth')