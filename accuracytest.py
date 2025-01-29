import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from mtcnn import MTCNN
import cv2

########################################
# User-Defined Configuration
########################################

data_dir = 'archive\\lfw-deepfunneled\\lfw-deepfunneled'  # Path to your LFW dataset
model_path = 'test.weights.h5'  # Path to your trained weights
input_shape = (224, 224, 3)        # Input shape used during training
num_same_pairs = 100
num_diff_pairs = 100

########################################
# Functions for Dataset Loading & Preprocessing
########################################

detector = MTCNN()

def load_dataset(data_dir, min_images_per_person=2):
    images = []
    labels = []
    label_dict = {}
    current_label = 0
    
    all_persons = os.listdir(data_dir)
    selected_persons = all_persons
    for person_name in selected_persons:
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        image_files = [f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))]
        if len(image_files) < min_images_per_person:
            continue
        label_dict[current_label] = person_name
        for image_name in image_files:
            image_path = os.path.join(person_dir, image_name)
            img = cv2.imread(image_path)
            if img is not None:
                images.append(img)
                labels.append(current_label)
        current_label += 1
    return images, np.array(labels), label_dict

def preprocess_faces(images, labels):
    processed_images = []
    processed_labels = []
    for idx, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)
        if results:
            # Find largest face
            max_area = 0
            largest_face = None
            for result in results:
                x, y, width, height = result['box']
                area = width * height
                if area > max_area:
                    max_area = area
                    largest_face = result['box']
            if largest_face is not None:
                x, y, width, height = largest_face
                h, w, _ = img_rgb.shape
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x1 + width)
                y2 = min(h, y1 + height)
                face = img_rgb[y1:y2, x1:x2]
                # Resize face to input_shape
                face = cv2.resize(face, (input_shape[0], input_shape[1]))
                processed_images.append(face)
                processed_labels.append(labels[idx])
    return np.array(processed_images), np.array(processed_labels)

########################################
# Model Creation
########################################

def create_embedding_model(input_shape):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    # Freeze base_model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)
    x = Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

def get_embeddings(model, images):
    images = images.astype('float32')
    images = preprocess_input(images)
    return model.predict(images)

########################################
# Distance Computation
########################################

def compute_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)

########################################
# Functions to Get Distances for Same/Diff Pairs
########################################

def get_same_diff_distances(embeddings, labels, num_same_pairs=100, num_diff_pairs=100):
    unique_labels = np.unique(labels)
    label_indices = {l: np.where(labels == l)[0] for l in unique_labels}
    
    same_dists = []
    diff_dists = []
    
    # Sample same-person pairs
    for _ in range(num_same_pairs):
        chosen_label = np.random.choice(unique_labels)
        idx = label_indices[chosen_label]
        if len(idx) < 2:
            continue
        i1, i2 = np.random.choice(idx, size=2, replace=False)
        dist = compute_distance(embeddings[i1], embeddings[i2])
        same_dists.append(dist)
    
    # Sample different-person pairs
    if len(unique_labels) > 1:
        for _ in range(num_diff_pairs):
            l1, l2 = np.random.choice(unique_labels, size=2, replace=False)
            idx1 = label_indices[l1]
            idx2 = label_indices[l2]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            i1 = np.random.choice(idx1)
            i2 = np.random.choice(idx2)
            dist = compute_distance(embeddings[i1], embeddings[i2])
            diff_dists.append(dist)
    
    return same_dists, diff_dists

def compute_accuracy_for_threshold(same_dists, diff_dists, threshold):
    # Correct same-person pairs are those with dist < threshold
    same_correct = sum(d < threshold for d in same_dists)
    # Correct different-person pairs are those with dist > threshold
    diff_correct = sum(d > threshold for d in diff_dists)
    total_pairs = len(same_dists) + len(diff_dists)
    if total_pairs == 0:
        return 0.0
    accuracy = (same_correct + diff_correct) / total_pairs
    return accuracy

########################################
# Main Execution
########################################

if __name__ == '__main__':
    # Load dataset
    images, labels, label_dict = load_dataset(data_dir, min_images_per_person=2)
    processed_images, processed_labels = preprocess_faces(images, labels)
    print(f"Processed images shape: {processed_images.shape}")
    print(f"Number of classes: {len(np.unique(processed_labels))}")
    
    # Create embedding model
    embedding_model = create_embedding_model(input_shape)
    # Load trained weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    embedding_model.load_weights(model_path)
    print("Model weights loaded successfully.")
    
    # Compute embeddings
    embeddings = get_embeddings(embedding_model, processed_images)
    
    # Get same/diff distances
    same_dists, diff_dists = get_same_diff_distances(embeddings, processed_labels, num_same_pairs, num_diff_pairs)
    
    if len(same_dists) == 0 or len(diff_dists) == 0:
        print("Not enough pairs to determine best threshold.")
        exit(0)
    
    # Combine all distances to determine candidate thresholds
    # We can consider thresholds at points between the distances or just use these distances as candidates
    all_dists = np.concatenate([same_dists, diff_dists])
    candidate_thresholds = np.unique(all_dists)
    
    best_threshold = None
    best_accuracy = -1.0
    
    # Evaluate accuracy for each candidate threshold
    for th in candidate_thresholds:
        acc = compute_accuracy_for_threshold(same_dists, diff_dists, th)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = th
    
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
