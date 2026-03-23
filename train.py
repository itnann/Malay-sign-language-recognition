import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from model import CustomLSTM
from data_process import gestures

X = np.load('X_TRAIN_2.npy')
y = np.load('y_TRAIN_2.npy')

print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert data to PyTorch tensors and move to the GPU
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long)  # Convert to class indices
y_test = torch.tensor(y_test, dtype=torch.long).to(device)  # Convert to class indices

batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create empty lists to store loss and accuracy values
train_losses = []
test_losses = []
test_accuracies = []

input_size = 258
hidden_size = 64
num_classes = len(gestures)
model = CustomLSTM(input_size, hidden_size, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Train the model on the GPU
num_epochs = 200
loss_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for X_train, y_train in train_loader:
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = model(X_train)  # Forward pass
        loss = criterion(outputs, y_train)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        epoch_loss += loss.item()  # Accumulate loss

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        accuracy = (test_outputs.argmax(dim=1) == y_test).float().mean()
    if (epoch + 1) % 10 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_loss:.4f} | Test Loss: {test_loss.item():.4f} | Test Acc: {accuracy.item():.4f}')

    if accuracy > best_test_acc:
        best_test_acc = accuracy
        torch.save(model.state_dict(), 'best_model.pth')

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
