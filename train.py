# train.py
from model import create_model
from load_data import load_and_preprocess_data
from train_test_split import split_data

def train_model(file_path):
    # Load and preprocess data
    data, num_users, num_items = load_and_preprocess_data(file_path)
    
    # Split data
    X_train, y_train, X_test, y_test = split_data(data)
    
    # Create model
    model = create_model(num_users, num_items)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.2, verbose=1)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    
    # Save the model
    model.save('ancf_model.h5')
    print("Model saved to ancf_model.h5")
    
    return model, history

if __name__ == "__main__":
    file_path = 'data.csv'
    model, history = train_model(file_path)
