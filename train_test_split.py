import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data):
    # Split data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Prepare data for model
    X_train = [train_data['CustomerID'].values, train_data['StockCode'].values]
    y_train = train_data['Rating'].values
    X_test = [test_data['CustomerID'].values, test_data['StockCode'].values]
    y_test = test_data['Rating'].values
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    from load_data import load_and_preprocess_data
    
    file_path = 'data.csv'
    data, num_users, num_items = load_and_preprocess_data(file_path)
    X_train, y_train, X_test, y_test = split_data(data)
    print(X_train[0][:5], y_train[:5])
    print(X_test[0][:5], y_test[:5])
