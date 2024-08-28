import numpy as np

# Load the processed data
user_item_data = np.load('user_item_ratings.npy')

# Check how many times -1.0 appears as a user_id
anomalous_count = np.sum(user_item_data[:, 0] == -1.0)
total_count = user_item_data.shape[0]

print(f"Occurrences of user_id = -1.0: {anomalous_count} out of {total_count} total records")

# Check the distribution of items for user_id = -1.0
anomalous_items = user_item_data[user_item_data[:, 0] == -1.0, 1]

# Find unique items and their counts
unique_items, counts = np.unique(anomalous_items, return_counts=True)

# Summarize the findings
total_unique_items = len(unique_items)
items_with_multiple_counts = np.sum(counts > 1)
top_5_items = sorted(zip(unique_items, counts), key=lambda x: x[1], reverse=True)[:5]

print(f"Total unique items for user_id = -1.0: {total_unique_items}")
print(f"Items that appear more than once for user_id = -1.0: {items_with_multiple_counts}")

print("\nTop 5 most frequent items for user_id = -1.0:")
for item, count in top_5_items:
    print(f"Item ID: {item}, Count: {count}")
