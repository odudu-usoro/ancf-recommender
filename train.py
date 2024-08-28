import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import create_ancf_model

# Load the data
user_item_data = np.load('user_item_ratings.npy')  # Replace with your actual data loading code
X = user_item_data[:, :2]  # First two columns: UserID and ItemID
y = user_item_data[:, 2]   # Third column: Rating

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Number of users and items
num_users = int(np.max(X[:, 0]) + 1)  # Assuming user IDs start from 0
num_items = int(np.max(X[:, 1]) + 1)  # Assuming item IDs start from 0

# Create the model
model = create_ancf_model(num_users, num_items)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    [X_train[:, 0], X_train[:, 1]],
    y_train,
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
    epochs=10,
    batch_size=256,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model on the test set
test_loss, test_mse = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
print(f"Test Loss: {test_loss}")
print(f"Test MSE: {test_mse}")
