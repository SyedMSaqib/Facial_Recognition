import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier




# Initialize MTCNN for face detection
detector = MTCNN()

# Load MobileNetV2 for feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Function to preprocess images
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return img

# Function to extract embeddings
def extract_embeddings(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    return base_model.predict(img).flatten()


def load_images_and_compute_embeddings():
    training_folder = 'training_data'
    labels = []
    embeddings = []
    
    for person_name in os.listdir(training_folder):
        person_folder = os.path.join(training_folder, person_name)
        
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            img = cv2.imread(image_path)
            
            # Detect faces using MTCNN
            faces = detector.detect_faces(img)
            
            for face in faces:
                x, y, w, h = face['box']
                face_img = img[y:y+h, x:x+w]
                
                # Extract embeddings
                embedding = extract_embeddings(face_img)
                embeddings.append(embedding)
                labels.append(person_name)
    
    return embeddings, labels

embeddings, labels = load_images_and_compute_embeddings()

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train a classifier (K-Nearest Neighbors) on embeddings
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(embeddings, encoded_labels)


def recognize_face(image_path):
    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)
    
    for face in faces:
        x, y, w, h = face['box']
        face_img = img[y:y+h, x:x+w]
        
        # Extract embeddings
        embedding = extract_embeddings(face_img)
        
        # Predict using classifier
        prediction = classifier.predict([embedding])[0]
        person_name = label_encoder.inverse_transform([prediction])[0]
        
        # Draw bounding box and label
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = 'henry.jpg'
recognize_face(image_path)
