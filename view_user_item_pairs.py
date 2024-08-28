import numpy as np

# Load the processed data
user_item_data = np.load('user_item_ratings.npy')

# Display the first 10 entries
for i in range(10):
    print(f"User ID: {user_item_data[i, 0]}, Item ID: {user_item_data[i, 1]}, Quantity: {user_item_data[i, 2]}")
