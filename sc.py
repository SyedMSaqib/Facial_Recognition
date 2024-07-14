import os
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms

# Load the MTCNN face detector
mtcnn = MTCNN(keep_all=False)

# Load the pretrained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define a normalization transformation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# Function to get embeddings from a cropped face
def get_embedding(model, image_path):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)
    if face is not None:
        face = normalize(face).unsqueeze(0)
        with torch.no_grad():
            embedding = model(face)
        return embedding
    else:
        return None

# Function to aggregate embeddings
def aggregate_embeddings(embeddings):
    return torch.mean(torch.stack(embeddings), dim=0, keepdim=True)

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
                embedding = get_embedding(model, image_path)
                if embedding is not None:
                    embeddings.append(embedding)
        if embeddings:
            user_embeddings[user_folder] = aggregate_embeddings(embeddings)

# Function to compare embeddings using cosine similarity
def compare_embeddings(embedding1, embedding2, threshold=0.3):
    cosine_similarity = F.cosine_similarity(embedding1, embedding2)
    avg_similarity = torch.mean(cosine_similarity).item()
    return avg_similarity > threshold

# Get embedding for the new image
# new_image_path = 'robert.jpg'
# new_image_path = '3.jpg'
new_image_path = 'both.jpg'
# new_image_path = '4.jpg'
new_embedding = get_embedding(model, new_image_path)

if new_embedding is not None:
    # Compare with each of the aggregated user embeddings and print the name
    for name, embedding in user_embeddings.items():
        if compare_embeddings(new_embedding, embedding):
            print(f"The person is: {name}")
            break
    else:
        print("No match found")
else:
    print("No face detected in the new image")
