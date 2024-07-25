import os
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.preprocessing import normalize as sknormalize
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
    boxes = []
    if faces is not None:
        for i, face in enumerate(faces):
            # Apply normalization and reshape for the model input
            face = normalize(face.unsqueeze(0))  # Add batch dimension and normalize
            with torch.no_grad():
                embedding = model(face)  # Assuming model output is a tensor
            embeddings.append(embedding.squeeze().numpy())  # Convert to numpy arrays
            boxes.append(mtcnn.detect(image)[0][i])  # Store the bounding box coordinates
    return embeddings, boxes

# Function to compute cosine similarity
def cosine_similarity(embedding1, embedding2):
    embedding1 = sknormalize(embedding1.reshape(1, -1))
    embedding2 = sknormalize(embedding2.reshape(1, -1))
    return np.dot(embedding1, embedding2.T)

# Generate embedding for the target image
def get_target_embedding(image_path, model):
    img = Image.open(image_path).convert('RGB')
    embeddings, _ = get_embeddings(model, img)
    if embeddings:
        return embeddings[0]
    else:
        raise ValueError("No face detected in the target image.")

# Function to process video and identify the target person
def process_video(video_path, output_path, model, target_embedding, threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # Convert the frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Get embeddings for the current frame
        embeddings, boxes = get_embeddings(model, pil_image)
        
        if embeddings:
            for idx, embedding in enumerate(embeddings):
                similarity = cosine_similarity(embedding, target_embedding)[0][0]

                box = boxes[idx]
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                if similarity > threshold:
                    label = f"Match: {similarity:.2f}"
                    color = (0, 255, 0)  # Green for recognized faces
                else:
                    label = "No match"
                    color = (0, 0, 255)  # Red for unrecognized faces
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Write the frame with rectangles and labels to the output video
        out.write(frame)

    cap.release()
    out.release()

# Path to the target image
target_image_path = '1m.jpg'

# Generate the target embedding
target_embedding = get_target_embedding(target_image_path, model)

# Path to the video file
video_path = 'videos/mark.mp4'
output_path = 'videos/output.mp4'

# Process the video to find the target person
process_video(video_path, output_path, model, target_embedding)
