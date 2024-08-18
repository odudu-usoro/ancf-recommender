# model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Multiply, Concatenate, Dropout
from tensorflow.keras.models import Model

def create_model(num_users, num_items, embedding_dim=50):
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    
    # Embedding layers
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)
    
    # Flatten embeddings
    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)
    
    # Interaction layer: element-wise product
    interaction = Multiply()([user_vec, item_vec])
    
    # Attention mechanism
    attention = Dense(1, activation='softmax')(interaction)
    weighted_interaction = Multiply()([interaction, attention])
    
    # Concatenate interaction and embeddings
    interaction_concat = Concatenate()([user_vec, item_vec, weighted_interaction])
    
    # Fully connected layers for prediction
    x = Dense(128, activation='relu')(interaction_concat)
    x = Dropout(0.2)(x)  # Dropout for regularization
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    # Model definition
    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    from load_data import load_and_preprocess_data
    from train_test_split import split_data
    
    file_path = 'data.csv'
    data, num_users, num_items = load_and_preprocess_data(file_path)
    X_train, y_train, X_test, y_test = split_data(data)
    
    model = create_model(num_users, num_items)
    model.summary()
