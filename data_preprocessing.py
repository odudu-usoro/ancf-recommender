import numpy as np
import pandas as pd

def preprocess_data(csv_file):
    # Open the file with 'ascii' encoding and ignore errors, then pass to pandas
    with open(csv_file, 'r', encoding='ascii', errors='ignore') as file:
        df = pd.read_csv(file)
    
    # Print columns to ensure we are working with the correct dataset
    print("Columns in the dataset:", df.columns)

    # Filter out records with user_id = -1.0 and assign a separate class label
    placeholder_data = df[df['CustomerID'] == -1.0]
    regular_data = df[df['CustomerID'] != -1.0]
    
    # Assign a unique user ID for the placeholder group (e.g., max regular ID + 1)
    max_regular_id = regular_data['CustomerID'].astype('category').cat.codes.max()
    placeholder_data['CustomerID'] = max_regular_id + 1
    
    # Combine the datasets back together
    df = pd.concat([regular_data, placeholder_data])
    
    # Use Quantity as the rating-like feature and normalize it
    df['Quantity'] = df['Quantity'] / df['Quantity'].max()

    # Convert categorical data into numerical data (e.g., CustomerID/StockCode to indices)
    df['CustomerID'] = df['CustomerID'].astype('category').cat.codes
    df['StockCode'] = df['StockCode'].astype('category').cat.codes

    # Convert DataFrame to NumPy array
    user_item_ratings = df[['CustomerID', 'StockCode', 'Quantity']].to_numpy()

    # Save processed data
    np.save('user_item_ratings.npy', user_item_ratings)
    print("Data preprocessing complete. Data saved to 'user_item_ratings.npy'.")

# Call the preprocessing function
preprocess_data('data.csv')
