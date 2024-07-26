import os
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine
import cv2

def process_video(video_path, training_folders):

    mtcnn = MTCNN(keep_all=True)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def get_embeddings(model, image):
        faces = mtcnn(image)
        embeddings = []
        boxes = []
        if faces is not None:
            for i, face in enumerate(faces):
                face = normalize(face.unsqueeze(0))
                with torch.no_grad():
                    embedding = model(face)
                embeddings.append(embedding.squeeze().numpy())
                boxes.append(mtcnn.detect(image)[0][i])
        return embeddings, boxes

    training_folder = training_folders

    user_embeddings = []
    user_labels = []

    for user_folder in os.listdir(training_folder):
        user_folder_path = os.path.join(training_folder, user_folder)
        if os.path.isdir(user_folder_path):
            for file_name in os.listdir(user_folder_path):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(user_folder_path, file_name)
                    img = Image.open(image_path).convert('RGB')
                    embeddings, _ = get_embeddings(model, img)
                    if embeddings:
                        for embedding in embeddings:
                            user_embeddings.append(embedding)
                            user_labels.append(user_folder)

    user_embeddings = np.array(user_embeddings)
    user_labels = np.array(user_labels)

    label_encoder = LabelEncoder()
    user_labels_encoded = label_encoder.fit_transform(user_labels)

    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(user_embeddings, user_labels_encoded)

    cap = cv2.VideoCapture(video_path)

    total_faces = 0
    total_recognized_faces = 0
    recognized_names = set()
    unique_embeddings = []

    threshold = 0.9 

    def is_new_face(embedding, unique_embeddings, threshold):
        for existing_embedding in unique_embeddings:
            if cosine(embedding.flatten(), existing_embedding.flatten()) < threshold:
                return False
        return True

    face_trackers = []
    max_distance = 50  # Maximum distance in pixels to consider the same face

    frame_skip = 100  
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embeddings, boxes = get_embeddings(model, pil_image)

        if embeddings:
            for idx, embedding in enumerate(embeddings):
                embedding = np.array(embedding).reshape(-1)
                predictions = svm_classifier.predict_proba(embedding.reshape(1, -1))
                best_match_idx = np.argmax(predictions, axis=1)
                best_match_label = label_encoder.inverse_transform(best_match_idx)
                best_match_confidence = predictions[0, best_match_idx][0]
                box = boxes[idx]

                x1, y1, x2, y2 = [int(coord) for coord in box]
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                matched_tracker = None
                for tracker in face_trackers:
                    tracker_x, tracker_y = tracker['position']
                    distance = np.sqrt((center_x - tracker_x)**2 + (center_y - tracker_y)**2)
                    if distance < max_distance:
                        matched_tracker = tracker
                        break

                if matched_tracker:
                    matched_tracker['position'] = (center_x, center_y)
                else:
                    face_trackers.append({'position': (center_x, center_y)})
                    total_faces += 1

                    if best_match_confidence > 0.9:
                        if is_new_face(embedding, unique_embeddings, threshold):
                            unique_embeddings.append(embedding)
                            total_recognized_faces += 1
                            recognized_names.add(best_match_label[0])

    cap.release()

    print(f"Total number of faces detected: {total_faces}")
    print(f"Total number of recognized faces: {total_recognized_faces}")
    print(f"Names of recognized faces: {', '.join(recognized_names)}")

video_path = 'videos/presidentTrumpInUN.mp4'
images = 'training_data/'
process_video(video_path, images)
