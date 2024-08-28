import pandas as pd

def load_and_preprocess_data(file_path):
    # Load the dataset with 'latin1' encoding
    dataset = pd.read_csv(file_path, encoding='latin1')
    
    # Aggregate data to get the user-item interactions
    user_item_data = dataset.groupby(['CustomerID', 'StockCode']).size().reset_index(name='Rating')
    
    # Remove rows with missing CustomerID
    user_item_data = user_item_data.dropna(subset=['CustomerID'])
    
    # Map user IDs and item IDs to unique indices
    user_ids = user_item_data['CustomerID'].unique()
    item_ids = user_item_data['StockCode'].unique()
    user_map = {user: i for i, user in enumerate(user_ids)}
    item_map = {item: i for i, item in enumerate(item_ids)}
    
    user_item_data['CustomerID'] = user_item_data['CustomerID'].map(user_map)
    user_item_data['StockCode'] = user_item_data['StockCode'].map(item_map)
    
    return user_item_data, len(user_ids), len(item_ids)

if __name__ == "__main__":
    file_path = 'data.csv'
    data, num_users, num_items = load_and_preprocess_data(file_path)
    print(data.head())
    print(f'Number of users: {num_users}, Number of items: {num_items}')
