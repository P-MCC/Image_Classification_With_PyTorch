import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from CustomDataLoader import PlayingCardDataset
from Model import AlexNet
from torchvision import transforms
import numpy as np

# Load the trained model
model_path = './Models/AlexNet_2024-11-15_18-16-49.pth'  # Replace this with the path to your saved model
num_classes = len(PlayingCardDataset(data_dir='M:/Datasets/Playing_Card/train').classes)
model = AlexNet(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Move model to device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare the test dataset
data_path = 'M:/Datasets/Playing_Card'
test_dir = '/test'
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # Resize images to 227x227 pixels
    transforms.ToTensor()           # Convert images to PyTorch tensors
])
test_dataset = PlayingCardDataset(data_dir=data_path + test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Test the model
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f'Test Accuracy: {accuracy:.4f}')

# Display confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
print('Classification Report:')
print(class_report)
