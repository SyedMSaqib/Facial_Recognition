import os
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import cv2

# Load the MTCNN face detector
mtcnn = MTCNN(keep_all=True)  # Set keep_all=True to detect multiple faces

# Load the pretrained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define a normalization transformation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# Function to get embeddings from cropped faces
def get_embeddings(model, image):
    faces = mtcnn(image)
    embeddings = []
    if faces is not None:
        for face in faces:
            # Apply normalization and reshape for the model input
            face = normalize(face.unsqueeze(0))  # Add batch dimension and normalize
            with torch.no_grad():
                embedding = model(face)  # Assuming model output is a tensor
            embeddings.append(embedding.squeeze().numpy())  # Convert to numpy arrays
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
                img = Image.open(image_path).convert('RGB')
                embeddings = get_embeddings(model, img)
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

# Function to process video and identify faces
def process_video(video_path, model, svm_classifier, label_encoder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 != 0:  # Process every 10th frame to save time
            continue

        # Convert the frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Get embeddings for the current frame
        embeddings = get_embeddings(model, pil_image)
        
        if embeddings:
            for idx, embedding in enumerate(embeddings):
                embedding = np.array(embedding).reshape(1, -1)
                predictions = svm_classifier.predict_proba(embedding)
                best_match_idx = np.argmax(predictions, axis=1)
                best_match_label = label_encoder.inverse_transform(best_match_idx)
                best_match_confidence = predictions[0, best_match_idx][0]
                
                if best_match_confidence > 0.9:
                    print(f"Frame {frame_count}: Person {idx + 1} is {best_match_label[0]} with confidence {best_match_confidence:.2f}")
                else:
                    print(f"Frame {frame_count}: Person {idx + 1} is: No match found")
        else:
            print(f"Frame {frame_count}: No faces detected")

    cap.release()

# Path to the video file
video_path = 'videos/mark.mp4'
process_video(video_path, model, svm_classifier, label_encoder)
