import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from model import create_model

def create_model_directory(directory_path='models'):
    """
    This method creates a directory to store model checkpoints and final models.
    It checks if the directory exists, and if not, creates it.
    """
    os.makedirs(directory_path, exist_ok=True)
    print(f"Model directory is ready at: {directory_path}")

X = np.load('../data/processed/X.npy')  
y = np.load('../data/processed/y.npy')  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

input_size = 13 
hidden_size = 256  
num_classes = 10 
learning_rate = 0.001
epochs = 50
dropout = 0.5 

model = create_model(input_size, hidden_size, num_classes).to(device)  
criterion = torch.nn.CrossEntropyLoss() 
optimizer = optim.AdamW(model.parameters(), lr=learning_rate) 

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

patience = 5  
patience_counter = 0
best_val_acc = 0  

create_model_directory('models')

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []


for epoch in range(epochs):
    model.train() 
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        max_grad_norm = 5.0  
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    model.eval() 
    val_loss = 0
    val_correct = 0
    val_total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad(): 
        for batch in val_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(y_batch.cpu().numpy())
    
    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_accuracies.append(100 * correct / total)
    val_accuracies.append(100 * val_correct / val_total)
    
    scheduler.step(val_loss)
    
    val_accuracy = 100 * val_correct / val_total
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), '../models/speech_recognition_best.pth')
        patience_counter = 0  
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break
    
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {total_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {100 * correct / total:.2f}%, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Accuracy: {100 * val_correct / val_total:.2f}%")
    
    print(classification_report(true_labels, predictions))

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'../models/speech_recognition_epoch_{epoch+1}.pth')

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.show()

plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

torch.save(model.state_dict(), '../models/speech_recognition_final.pth')
