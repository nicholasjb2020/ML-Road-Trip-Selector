import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical, split_dataset
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import random

theme_names = ['Space', 'Wildlife', 'Festivals', 'History', 'Museums', 'Romantic', 'Nightlife', 'Architecture', 'Art', 'Shopping']

def get_user_input():
    user_preferences = []
    print('Please rate on a scale of 1-10, how much you enjoy the following theme.\n')
    for i in range(0, 10):
       user_preferences.append(input(f'{theme_names[i]}:'))
    return user_preferences

def evaluate_model(model, X, y, k=5):
    """
    Evaluate the accuracy of a model using k-fold cross-validation.

    Parameters:
    - model: The Keras model to be evaluated.
    - X: The input data.
    - y: The true labels.
    - k: The number of folds for cross-validation (default is 5).

    Returns:
    - mean_accuracy: The mean accuracy over k folds.
    """
    # Convert labels to one-hot encoding if needed
    y_categorical = to_categorical(y)

    # Function to calculate accuracy using cross_val_score
    def get_accuracy(estimator):
        _, accuracy = estimator.evaluate(X, y_categorical, verbose=0)
        return accuracy

    keras_model = KerasClassifier(build_fn=lambda: model, epochs=1, batch_size=32, verbose=0)

    scores = cross_val_score(keras_model, X, y, cv=k, scoring=get_accuracy)

    mean_accuracy = np.mean(scores)

    print(f'Mean Accuracy: {mean_accuracy * 100:.2f}%')

    return mean_accuracy

def train_network(training_data):
    X = training_data.iloc[:, 1:]
    y = training_data.iloc[:, 0]

    model = Sequential()
    model.add(Dense(units=10, input_dim=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    evaluate_model(model, X, y, k=5)

    return model

def classify_user_preferences(model, user_preferences):
    """
    Classify user preferences using the trained model.

    Parameters:
    - model: The trained Keras model.
    - user_preferences: A list of length 10 representing user preferences.

    Returns:
    - predicted_theme: The predicted theme based on the model.
    """
    user_input = np.array(user_preferences).reshape(1, -1)
    predictions = model.predict(user_input)

    predicted_class_index = np.argmax(predictions)
    predicted_theme = theme_names[predicted_class_index]

    return predicted_theme

def get_random_input_from_cluster(predicted_theme, training_data):
    """
    Get a random input from the training data with the predicted theme as its target.

    Parameters:
    - predicted_theme: The predicted theme based on the user preferences.
    - training_data: The DataFrame containing the training data.

    Returns:
    - random_input: A random input from the cluster with the predicted theme.
    """
    # Filter training data based on the predicted theme
    cluster_data = training_data[training_data['utility'] == predicted_theme]

    # Get a random row from the filtered data
    random_input = cluster_data.sample(n=1, random_state=42)

    # Remove the target column (assuming 'target_column' is the name of your target column)
    random_input = random_input.drop('utility', axis=1)

    return random_input

data = pd.read_csv('data_files/new_data.csv')
user_pref = get_user_input()
trained_model = train_network(data)
