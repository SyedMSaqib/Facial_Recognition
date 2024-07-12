import os
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
import numpy as np

# Load the pretrained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define a transformation to resize and normalize the images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to get embeddings
def get_embedding(model, image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img)
    return embedding

# Folder containing training images
training_folder = 'training_data/'

# Generate embeddings for training images
embeddings = []
for user_folder in os.listdir(training_folder):
    user_folder_path = os.path.join(training_folder, user_folder)
    if os.path.isdir(user_folder_path):
        for file_name in os.listdir(user_folder_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(user_folder_path, file_name)
                embedding = get_embedding(model, image_path)
                embeddings.append((user_folder, embedding))

# Function to compare embeddings
def compare_embeddings(embedding1, embedding2, threshold=1.0):
    distance = F.pairwise_distance(embedding1, embedding2)
    return distance.item() < threshold

# Get embedding for the new image
new_image_path = '4.jpg'
new_embedding = get_embedding(model, new_image_path)

# Compare with each of the training embeddings and print the name
for name, embedding in embeddings:
    if compare_embeddings(new_embedding, embedding):
        print(f"The person is: {name}")
        break
else:
    print("No match found")
