import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

def create_ancf_model(num_users, num_items, embedding_dim=50):
    # User and item inputs
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    # Embedding layers for users and items
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)

    # Flatten the embeddings to 1D tensors
    user_vecs = Flatten()(user_embedding)
    item_vecs = Flatten()(item_embedding)

    # Element-wise multiplication of user and item vectors
    user_item_interaction = Multiply()([user_vecs, item_vecs])

    # Neural Network Layers
    concatenated = Concatenate()([user_vecs, item_vecs, user_item_interaction])
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='linear')(x)  # Using 'linear' activation for regression output

    # Creating the model
    model = Model(inputs=[user_input, item_input], outputs=output)

    # Compiling the model with Mean Squared Error loss function
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    model.summary()
    return model
