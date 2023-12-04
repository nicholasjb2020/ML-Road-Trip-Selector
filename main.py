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
    # user_preferences = []
    # print('Please rate on a scale of 1-10, how much you enjoy the following theme.\n')
    # for i in range(0, 10):
    #    user_preferences.append(float(input(f'{theme_names[i]}:')))
    return [1, 7, 2, 9, 6, 8, 5, 5, 2, 9]

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
                             verbose=1)
        # Save model trained on each fold.

        # Evaluate the model - report accuracy and capture it into a list for future reporting
        scores = model2.evaluate(test_X, Y_one_hot[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)

        fold_no = fold_no + 1

    mean_accuracy = sum(acc_per_fold) / k
    print(f'Mean Accuracy across {k} folds: {mean_accuracy:.2f}%')


def train_network(training_data, epochs=100, batch_size=8):
    X = training_data.iloc[:, 1:]
    y = training_data.iloc[:, 0]

    # Step 1: Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)

    # Step 2: Build the model
    model = Sequential()
    model.add(Dense(units=390, input_dim=10, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=75, activation='relu'))
    model.add(Dense(units=390, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Output layer should have the correct number of units based on the number of classes
    num_classes = len(label_encoder.classes_)
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Step 3: Train the model
    model.fit(X, y_one_hot, epochs=epochs, batch_size=batch_size, verbose=0)

    # Step 4: Return the trained model and label encoder
    return model, label_encoder


def classify_user_preferences(model, user_preferences, label_encoder, k=1):
    """
    Classify user preferences using the trained model and return the k-th best predicted theme.

    Parameters:
    - model: The trained Keras model.
    - user_preferences: A list of length 10 representing user preferences.
    - label_encoder: The label encoder used for encoding classes.
    - k: The position of the predicted theme to return (default is 1).

    Returns:
    - kth_best_theme: The k-th best predicted theme based on the model.
    """
    user_input = np.array(user_preferences).reshape(1, -1)
    predictions = model.predict(user_input)

    # Get the indices of the predicted classes sorted in descending order
    sorted_indices = np.argsort(predictions[0])[::-1]

    # Get the k-th best index
    kth_best_index = sorted_indices[k - 1]

    # Decode the label using the label encoder
    kth_best_theme = label_encoder.classes_[kth_best_index]

    return kth_best_theme


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

data = pd.read_csv('data_files/clustered_data.csv')
user_pref = get_user_input()
model, encoder = train_network(data)
prediction = classify_user_preferences(model, user_pref, encoder)
print(get_random_input_from_cluster(prediction, data))
