import os
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
import numpy as np

# Load the MTCNN face detector
mtcnn = MTCNN(keep_all=True)  # Set keep_all=True to detect multiple faces

# Load the pretrained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define a normalization transformation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# Function to get embeddings from cropped faces
def get_embeddings(model, image_path):
    img = Image.open(image_path).convert('RGB')
    faces = mtcnn(img)
    embeddings = []
    if faces is not None:
        for face in faces:
            # Apply normalization and reshape for the model input
            face = normalize(face.unsqueeze(0))  # Add batch dimension and normalize
            with torch.no_grad():
                embedding = model(face)  # Assuming model output is a tensor
            embeddings.append(embedding.squeeze().numpy())  # Convert to numpy array
    return embeddings

# Folder containing training images
training_folder = 'training_data/'

# Generate embeddings for training images
user_embeddings = {}
for user_folder in os.listdir(training_folder):
    user_folder_path = os.path.join(training_folder, user_folder)
    if os.path.isdir(user_folder_path):
        embeddings = []
        for file_name in os.listdir(user_folder_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(user_folder_path, file_name)
                embedding = get_embeddings(model, image_path)
                if embedding:
                    embeddings.extend(embedding)
        if embeddings:
            user_embeddings[user_folder] = embeddings

# Function to compare embeddings using cosine similarity
# Function to compare embeddings using Euclidean distance
# def compare_embeddings(embedding1, embedding2, threshold=0.9):
#     euclidean_distance = np.linalg.norm(embedding1 - embedding2)
#     return euclidean_distance < threshold

def compare_embeddings(embedding1, embedding2, threshold=0.7):
    cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cosine_similarity > threshold

# Get embeddings for the new image
new_image_path = 'hg2.jpg'
# new_image_path = 'both.jpg'
new_embeddings = get_embeddings(model, new_image_path)

if new_embeddings:
    print(f"Number of persons detected: {len(new_embeddings)}")
    for idx, new_embedding in enumerate(new_embeddings):
        match_found = False
        for name, embeddings in user_embeddings.items():
            for embedding in embeddings:
                if compare_embeddings(new_embedding, embedding):
                    print(f"Person {idx + 1} is: {name}")
                    match_found = True
                    break
            if match_found:
                break
        if not match_found:
            print(f"Person {idx + 1} is: No match found")
else:
    print("No faces detected in the new image")
