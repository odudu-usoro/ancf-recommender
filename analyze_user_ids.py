import numpy as np

# Load the processed data
user_item_data = np.load('user_item_ratings.npy')

# Extract all user_ids
user_ids = user_item_data[:, 0]

# Find unique user_ids and their counts
unique_user_ids, counts = np.unique(user_ids, return_counts=True)

# Print the number of unique user_ids
print(f"Total unique user_ids: {len(unique_user_ids)}")

# Display the first 10 unique user_ids and their counts
print("\nFirst 10 unique user_ids and their occurrence counts:")
for i in range(min(10, len(unique_user_ids))):
    print(f"User ID: {unique_user_ids[i]}, Count: {counts[i]}")
