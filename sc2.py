import os
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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
user_embeddings = []
user_labels = []

for user_folder in os.listdir(training_folder):
    user_folder_path = os.path.join(training_folder, user_folder)
    if os.path.isdir(user_folder_path):
        for file_name in os.listdir(user_folder_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(user_folder_path, file_name)
                embeddings = get_embeddings(model, image_path)
                if embeddings:
                    for embedding in embeddings:
                        user_embeddings.append(embedding)
                        user_labels.append(user_folder)

user_embeddings = np.array(user_embeddings)
user_labels = np.array(user_labels)

# Encode labels
label_encoder = LabelEncoder()
user_labels_encoded = label_encoder.fit_transform(user_labels)

# Train SVM classifier
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(user_embeddings, user_labels_encoded)

# Get embeddings for the new image
new_image_path = 'r1.jpg'
new_embeddings = get_embeddings(model, new_image_path)

if new_embeddings:
    print(f"Number of persons detected: {len(new_embeddings)}")
    for idx, new_embedding in enumerate(new_embeddings):
        new_embedding = np.array(new_embedding).reshape(1, -1)
        predictions = svm_classifier.predict_proba(new_embedding)
        best_match_idx = np.argmax(predictions, axis=1)
        best_match_label = label_encoder.inverse_transform(best_match_idx)
        best_match_confidence = predictions[0, best_match_idx][0]
        
        if best_match_confidence > 0.9:  
            print(f"Person {idx + 1} is: {best_match_label[0]} with confidence {best_match_confidence:.2f}")
        else:
            print(f"Person {idx + 1} is: No match found")
else:
    print("No faces detected in the new image")
