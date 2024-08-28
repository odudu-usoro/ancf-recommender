import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    # Load the main dataset
    user_item_data = np.load('user_item_ratings.npy')

    # Convert to DataFrame for easier handling
    user_item_df = pd.DataFrame(user_item_data, columns=['CustomerID', 'StockCode', 'Rating'])
    
    # Map user IDs and item IDs to unique indices (if not already done in preprocessing)
    user_ids = user_item_df['CustomerID'].unique()
    item_ids = user_item_df['StockCode'].unique()
    user_map = {user: i for i, user in enumerate(user_ids)}
    item_map = {item: i for i, item in enumerate(item_ids)}
    
    user_item_df['CustomerID'] = user_item_df['CustomerID'].map(user_map)
    user_item_df['StockCode'] = user_item_df['StockCode'].map(item_map)
    
    return user_item_df, len(user_ids), len(item_ids)

if __name__ == "__main__":
    file_path = 'data.csv'
    data, num_users, num_items = load_and_preprocess_data(file_path)
    print(data.head())
    print(f'Number of users: {num_users}, Number of items: {num_items}')
