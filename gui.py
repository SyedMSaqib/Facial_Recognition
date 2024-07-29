import os
import tkinter as tk
from tkinter import filedialog, ttk, Text
import threading
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine
import cv2

# Process video function
def process_video(video_path, training_folder, result_text, progress_var):
    mtcnn = MTCNN(keep_all=True)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def get_embeddings(model, image):
        faces = mtcnn(image)
        embeddings = []
        if faces is not None:
            for face in faces:
                face = normalize(face.unsqueeze(0))
                with torch.no_grad():
                    embedding = model(face)
                embeddings.append(embedding.squeeze().numpy())
        return embeddings

    user_embeddings = []
    user_labels = []

    # Debug: Print the training folder path
    print(f"Training folder: {training_folder}")

    subfolders = [f for f in os.listdir(training_folder) if os.path.isdir(os.path.join(training_folder, f))]
    total_subfolders = len(subfolders)

    for idx, user_folder in enumerate(subfolders):
        user_folder_path = os.path.join(training_folder, user_folder)
        image_files = [f for f in os.listdir(user_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            print(f"Processing folder: {user_folder}")
            
            for file_name in image_files:
                image_path = os.path.join(user_folder_path, file_name)
                print(f"Processing image: {image_path}")
                img = Image.open(image_path).convert('RGB')
                embeddings = get_embeddings(model, img)
                if embeddings:
                    for embedding in embeddings:
                        user_embeddings.append(embedding)
                        user_labels.append(user_folder)
                else:
                    print(f"No embeddings found for image: {image_path}")

        # Update progress bar
        progress_var.set((idx + 1) / total_subfolders * 100)
        result_text.update_idletasks()

    if len(user_embeddings) == 0:
        return

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

    frame_skip = 10  # Skip every 10 frames
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embeddings = get_embeddings(model, pil_image)

        if embeddings:
            for embedding in embeddings:
                embedding = np.array(embedding).reshape(-1)
                if is_new_face(embedding, unique_embeddings, threshold):
                    unique_embeddings.append(embedding)
                    total_faces += 1

                predictions = svm_classifier.predict_proba(embedding.reshape(1, -1))
                best_match_idx = np.argmax(predictions, axis=1)
                best_match_label = label_encoder.inverse_transform(best_match_idx)
                best_match_confidence = predictions[0, best_match_idx][0]

                if best_match_confidence > 0.8:
                    recognized_names.add(best_match_label[0])
                    total_recognized_faces = len(recognized_names)

    cap.release()

    result_text.insert(tk.END, f"Total number of faces detected: {total_faces}\n")
    result_text.insert(tk.END, f"Total number of recognized faces: {total_recognized_faces}\n")
    result_text.insert(tk.END, f"Names of recognized faces: {', '.join(recognized_names)}\n")

# GUI code
def browse_image_directory():
    directory = filedialog.askdirectory()
    image_dir_entry.delete(0, tk.END)
    image_dir_entry.insert(0, directory)

def browse_video_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    video_file_entry.delete(0, tk.END)
    video_file_entry.insert(0, file_path)

def run_script():
    image_dir = image_dir_entry.get()
    video_file = video_file_entry.get()
    if not image_dir or not video_file:
        result_text.insert(tk.END, "Please select both image directory and video file.\n")
        return

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Processing...\n")
    run_button.config(state=tk.DISABLED)
    progress_var.set(0)

    def process_and_update():
        process_video(video_file, image_dir, result_text, progress_var)
        result_text.insert(tk.END, "Processing complete.\n")
        run_button.config(state=tk.NORMAL)

    threading.Thread(target=process_and_update).start()

app = tk.Tk()
app.title("Face Recognition in Video")
app.geometry("600x400")

style = ttk.Style(app)
style.configure("TButton", padding=6, relief="flat", background="#ccc")
style.configure("TLabel", padding=6, relief="flat", background="#eee")
style.configure("TEntry", padding=6, relief="flat", background="#fff")

# Frame for inputs
frame_inputs = ttk.Frame(app, padding="10")
frame_inputs.pack(fill=tk.X)

image_dir_label = ttk.Label(frame_inputs, text="Image Directory:")
image_dir_label.grid(row=0, column=0, sticky=tk.W, pady=5)
image_dir_entry = ttk.Entry(frame_inputs, width=40)
image_dir_entry.grid(row=0, column=1, pady=5)
image_dir_button = ttk.Button(frame_inputs, text="Browse", command=browse_image_directory)
image_dir_button.grid(row=0, column=2, pady=5, padx=5)

video_file_label = ttk.Label(frame_inputs, text="Video File:")
video_file_label.grid(row=1, column=0, sticky=tk.W, pady=5)
video_file_entry = ttk.Entry(frame_inputs, width=40)
video_file_entry.grid(row=1, column=1, pady=5)
video_file_button = ttk.Button(frame_inputs, text="Browse", command=browse_video_file)
video_file_button.grid(row=1, column=2, pady=5, padx=5)

run_button = ttk.Button(frame_inputs, text="Run Script", command=run_script)
run_button.grid(row=2, column=1, pady=10)

# Frame for results
frame_results = ttk.Frame(app, padding="10")
frame_results.pack(fill=tk.BOTH, expand=True)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(frame_results, variable=progress_var, maximum=100)
progress_bar.pack(fill=tk.X, pady=5)

result_text = Text(frame_results, height=15, wrap="word")
result_text.pack(fill=tk.BOTH, expand=True)

app.mainloop()
