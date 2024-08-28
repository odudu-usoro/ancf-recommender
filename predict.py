# predict.py
import numpy as np
import tensorflow as tf
from load_data import load_and_preprocess_data

def load_model(file_path):
    model = tf.keras.models.load_model(file_path)
    print("Model loaded from", file_path)
    return model

def make_prediction(model, user_id, item_id):
    # Prepare input data
    user_input = np.array([user_id])
    item_input = np.array([item_id])
    
    # Make prediction
    prediction = model.predict([user_input, item_input])
    return prediction

if __name__ == "__main__":
    model_path = 'ancf_model.h5'
    model = load_model(model_path)
    
    # Example prediction
    user_id = 540.0
    item_id = 3305.0
    prediction = make_prediction(model, user_id, item_id)
    print(f"Predicted rating for user {user_id} and item {item_id}: {prediction}")
