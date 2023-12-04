import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical, split_dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import random

theme_names = ['Space', 'Wildlife', 'Festivals', 'History', 'Museums', 'Romantic', 'Nightlife', 'Architecture', 'Art', 'Shopping']

def get_user_input():
    user_preferences = []
    print('Please rate on a scale of 1-10, how much you enjoy the following theme.\n')
    for i in range(0, 10):
       user_preferences.append(float(input(f'{theme_names[i]}:')))
    print('Recommending you a new road trip...\n')
    return user_preferences

def evaluate_model(model, X, Y, k=5):
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
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    Y_one_hot = to_categorical(Y_encoded)


    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []  # save accuracy from each fold
    X_array = X.to_numpy()

    for train, test in cv.split(X_array, Y):
        print('   ')
        print(f'Training for fold {fold_no} ...')

        # Scale data
        scaler = MinMaxScaler()
        train_X = X_array[train]
        test_X = X_array[test]
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        # Define the model - inside the loop so it trains from scratch for each fold
        # If defined outside, each fold training starts at where it left off at the previous fold
        # calling it as model2 instead of model to make sure no information from our
        # previous example is carried over (without restarting the kernel)
        model2 = Sequential()
        model2.add(Dense(units=390, input_dim=10, activation='relu'))
        model2.add(Dense(units=50, activation='relu'))
        model2.add(Dense(units=75, activation='relu'))
        model2.add(Dense(units=390, activation='softmax'))
        model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Fit data to model
        history = model2.fit(train_X, Y_one_hot[train],
                             batch_size=8,
                             epochs=100,
                             verbose=0)
        # Save model trained on each fold.

        # Evaluate the model - report accuracy and capture it into a list for future reporting
        scores = model2.evaluate(test_X, Y_one_hot[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)

        fold_no = fold_no + 1

    mean_accuracy = sum(acc_per_fold) / k
    print(f'Mean Accuracy across {k} folds: {mean_accuracy:.2f}%')


def train_network(data):
    # Assuming data is a NumPy array where the first column is the target variable
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    # Convert target variable to one-hot encoding
    y_one_hot = to_categorical(y, num_classes=390)

    model = Sequential()
    model.add(Dense(units=390, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=75, activation='relu'))
    model.add(Dense(units=390, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the whole dataset
    model.fit(X, y_one_hot, epochs=25, batch_size=32, verbose=0)

    return model

def classify_user_preferences(model, user_pref):
    """
    Classify user preferences using the trained model.

    Parameters:
    - model: The trained Keras model.
    - user_pref: A list of length 10 representing user preferences.

    Returns:
    - predicted_class: The predicted class based on the model.
    """
    user_input = np.array(user_pref).reshape(1, -1)
    predictions = model.predict(user_input)

    predicted_class = np.argmax(predictions)

    return predicted_class


def get_random_input_from_cluster(predicted_theme, training_data, k=0):
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

    trip_from_cluster = cluster_data.iloc[k]
    return trip_from_cluster.iloc[1:]

def prediction_output(predicted_class, data):
    print("We recommend a road trip with the following feature scores:")
    road_trip = get_random_input_from_cluster(predicted_class, data)
    for theme, pred in zip(theme_names, road_trip):
        print(f'{theme}: {pred}')


data = pd.read_csv('data_files/clustered_data.csv')
user_pref = get_user_input()
model = train_network(data)
predicted_class = classify_user_preferences(model, user_pref)
prediction_output(predicted_class, data)
