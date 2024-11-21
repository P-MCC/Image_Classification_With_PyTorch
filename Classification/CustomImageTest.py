import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from Model import AlexNet  # Ensure this matches the location of your model class definition
from CustomDataLoader import PlayingCardDataset  # Ensure this matches the location of your custom dataset

# Load the trained model
model_path = './Models/AlexNet_2024-11-15_18-16-49.pth'  # Replace with the path to your saved model
data_path = 'M:/Datasets/Playing_Card/train'  # Path to training data to get class names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
num_classes = len(PlayingCardDataset(data_dir=data_path).classes)
model = AlexNet(num_classes=num_classes)
state_dict = torch.load(model_path, map_location=device, weights_only=True)  # Load only weights
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Define the transformation to be applied to the custom image
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # Resize to match training input size
    transforms.ToTensor()           # Convert to tensor
])

# Function to test a custom image
def predict_custom_image(model, image_path, transform, class_names):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
    
    # Display the image and the predicted class
    plt.imshow(image)
    plt.title(f'Predicted: {class_names[predicted_class.item()]}')
    plt.axis('off')
    plt.show()

    print(f'Predicted Class: {class_names[predicted_class.item()]}')

# Replace with the path to your custom image
image_path = './Dataset/9_of_diamonds.jpg'  # e.g., 'M:/Datasets/Playing_Card/test/custom_image.jpg'
class_names = PlayingCardDataset(data_dir=data_path).classes

# Run the prediction
predict_custom_image(model, image_path, transform, class_names)
