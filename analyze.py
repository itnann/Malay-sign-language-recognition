from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd  # Add this line
import numpy as np
from sklearn.model_selection import train_test_split
from model import CustomLSTM
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

 # Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gestures = ['abang', 'ada', 'ambil', 'anak_lelaki', 'anak_perempuan', 'apa', 'apa_khabar',
            'arah', 'assalamualaikum', 'ayah', 'baca', 'bagaimana', 'bahasa_isyarat',
            'baik', 'baik_2', 'bapa', 'bapa_saudara', 'bas', 'bawa', 'beli', 'beli_2',
            'berapa', 'berjalan', 'berlari', 'bila', 'bola', 'boleh', 'bomba', 'buang',
            'buat', 'curi', 'dapat', 'dari', 'emak', 'emak_saudara', 'hari', 'hi', 'hujan',
            'jahat', 'jam', 'jangan', 'jumpa', 'kacau', 'kakak', 'keluarga', 'kereta',
            'kesakitan', 'lelaki', 'lemak', 'lupa', 'main', 'makan', 'mana', 'marah', 'mari',
            'masa', 'masalah', 'minum', 'mohon', 'nasi', 'nasi_lemak', 'panas', 'panas_2',
            'pandai', 'pandai_2', 'payung', 'pen', 'pensil', 'perempuan', 'pergi', 'pergi_2',
            'perlahan', 'perlahan_2', 'pinjam', 'polis', 'pukul', 'ribut', 'sampai',
            'saudara', 'sejuk', 'sekolah', 'siapa', 'sudah', 'suka', 'tandas', 'tanya',
            'teh_tarik', 'teksi', 'tidur', 'tolong']

print(f'f{len(gestures)}')
X = np.load('X_TRAIN_2.npy')
y = np.load('y_TRAIN_2.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)  # Convert to class indices

model = CustomLSTM(258, 64, len(gestures)).to(device)
model.load_state_dict(torch.load('best_model.pth'))

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

# Evaluate the model on the GPU
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    accuracy = (test_outputs.argmax(dim=1) == y_test).float().mean()
    print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

    # Convert predictions and true labels to numpy arrays
    predicted_labels = test_outputs.argmax(dim=1).cpu().numpy()
    true_labels = y_test.cpu().numpy()

    # Calculate and print the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Calculate and print the classification report
    class_report = classification_report(true_labels, predicted_labels, target_names=gestures)
    print("Classification Report:")
    print(class_report)

    # Plot the classification report
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.DataFrame.from_dict(classification_report(true_labels, predicted_labels, target_names=gestures, output_dict=True)).T, annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.show()


