import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_dataset(csv_file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(csv_file_path)

def preprocess_image(image_path, target_size=(100, 100)):
    """Load and preprocess image: grayscale, resize, normalize."""
    image = load_img(image_path, color_mode='grayscale', target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

def build_siamese_model(input_shape):
    """Builds the base network for the Siamese model."""
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    output_layer = Dense(128, activation='relu')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)

def euclidean_distance(vectors):
    """Compute the Euclidean distance between feature vectors."""
    f1, f2 = vectors
    return K.sqrt(K.sum(K.square(f1 - f2), axis=1, keepdims=True))

def create_pairs(df, train_folder_path):
    """Creates image pairs and their corresponding labels from the CSV."""
    pairs = []
    labels = []
    
    for _, row in df.iterrows():
        img1 = preprocess_image(os.path.join(train_folder_path, row[0]))
        img2 = preprocess_image(os.path.join(train_folder_path, row[1]))
        pairs.append([img1, img2])
        labels.append(row[2])
        
    pairs = np.array(pairs)
    return [pairs[:, 0], pairs[:, 1]], np.array(labels)

def build_siamese_network(input_shape):
    """Builds the complete Siamese network with a distance computation layer."""
    base_network = build_siamese_model(input_shape)
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    
    model = Model([input_a, input_b], distance)
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])
    
    return model

def train_siamese_model(model, pairs, labels, epochs=10, batch_size=32):
    """Trains the Siamese model using the provided pairs and labels."""
    pairs[0] = np.expand_dims(pairs[0], axis=-1)
    pairs[1] = np.expand_dims(pairs[1], axis=-1)
    model.fit([pairs[0], pairs[1]], labels, epochs=epochs, batch_size=batch_size)

def verify_signature(model, test_image_path, reference_image_path, threshold=0.5):
    """Verifies if the test signature matches the reference signature."""
    test_image = preprocess_image(test_image_path)
    reference_image = preprocess_image(reference_image_path)
    
    test_image = np.expand_dims(np.expand_dims(test_image, axis=0), axis=-1)
    reference_image = np.expand_dims(np.expand_dims(reference_image, axis=0), axis=-1)
    
    dist = model.predict([test_image, reference_image])[0][0]
    
    if dist > threshold:
        return "Test Signature is authentic."
    else:
        return "Test Signature is a forgery."

if __name__ == "__main__":
    # Paths
    csv_file_path = r'C:\Users\shiva\OneDrive\Desktop\trans-trip\train_data.csv'
    train_folder_path = r'C:\Users\shiva\Downloads\nitk_intern\train'
    
    # Load dataset
    df = load_dataset(csv_file_path)
    
    # Define input shape
    input_shape = (100, 100, 1)
    
    # Build and compile the model
    siamese_model = build_siamese_network(input_shape)
    
    # Create pairs from the CSV
    pairs, labels = create_pairs(df, train_folder_path)
    
    # Train the model
    train_siamese_model(siamese_model, pairs, labels, epochs=10, batch_size=32)

from tensorflow.keras.models import load_model

# Load your trained model (assuming you have it in memory already)
trained_model = siamese_model  # Replace with your trained model variable

# Specify the path where you want to save the model
model_save_path = r'C:\Users\shiva\OneDrive\Desktop\trans-trip\siamese_model.h5'

# Save the trained model
trained_model.save(model_save_path)
print(f"Model saved to {model_save_path}")

