import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import KFold
import numpy as np

theme_names = ['Space', 'Wildlife', 'Festivals', 'History', 'Museums', 'Romantic', 'Nightlife', 'Architecture', 'Art', 'Shopping']

def get_user_input():
    """
    Get user preferences for road trip themes.

    This function prompts the user to rate their enjoyment of each road trip theme on a scale of 1 to 10. The user is
    asked to input a numerical rating for each theme, and the preferences are stored in a list.

    Returns:
    - user_preferences: A list containing user ratings for each road trip theme.
    """
    user_preferences = []
    print('Please rate on a scale of 1-10, how much you enjoy the following theme.\n')

    for i in range(0, 10):
        user_preferences.append(float(input(f'{theme_names[i]}:')))

    print('Recommending you a new road trip...\n')
    return user_preferences



def evaluate_model(data, k=10):
    """
    Evaluate a neural network model using k-fold cross-validation.

    Parameters:
    - data: A Pandas DataFrame containing the target variable in the first column and features in the remaining columns.
    - k: The number of folds for cross-validation (default is 10).

    This function performs k-fold cross-validation on a neural network model using the specified data. The input data is
    assumed to have a structure where the first column represents the target variable, and the rest of the columns
    represent features.

    The training is performed on each fold separately, and the mean accuracy across all folds is reported.

    """
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    Y_one_hot = to_categorical(Y, num_classes=390)

    # K-fold cross-validation setup
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []  # save accuracy from each fold
    X_array = X.to_numpy()

    print('Performing cross-validation...\n')
    for train, test in cv.split(X_array, Y):
        train_X = X_array[train]
        test_X = X_array[test]

        # Create a sequential neural network model
        model = Sequential()
        model.add(Dense(units=390, input_dim=10, activation='relu'))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dense(units=75, activation='relu'))
        model.add(Dense(units=390, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit data to model
        model.fit(train_X, Y_one_hot[train], batch_size=8, epochs=100, verbose=0)

        # Evaluate the model - report accuracy and capture it into a list for future reporting
        scores = model.evaluate(test_X, Y_one_hot[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)

        fold_no = fold_no + 1

    mean_accuracy = sum(acc_per_fold) / k
    print(f'Mean Accuracy across {k} folds: {mean_accuracy:.2f}%')


def train_network(data):
    """
    Train a neural network model using the specified data.

    Parameters:
    - data: A Pandas DataFrame containing the target variable in the first column and features in the remaining columns.

    Returns:
    - model: The trained Keras model.

    This function assumes that the input data has a structure where the first column represents the target variable
    and the rest of the columns represent features. The target variable is converted to one-hot encoding, and a neural
    network model is created and trained using Keras.

    The architecture of the neural network:
    - Input layer with units equal to the number of features.
    - Hidden layers with 390, 50, and 75 units, respectively, and ReLU activation.
    - Output layer with units equal to the number of classes (390) and softmax activation for multi-class
    classification.

    The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation
     metric.

    The training is performed on the entire dataset for a specified number of epochs (default is 25) with a batch size
     of 32.

    Uncomment the line '#evaluate_model(data)' to perform k-fold cross-validation on the model.

    """
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    # Convert target variable to one-hot encoding
    y_one_hot = to_categorical(y, num_classes=390)

    # Create a sequential neural network model
    model = Sequential()
    model.add(Dense(units=390, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=75, activation='relu'))
    model.add(Dense(units=390, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the whole dataset
    model.fit(X, y_one_hot, epochs=25, batch_size=32, verbose=0)

    # Uncomment this line to perform k-fold cross-validation on the model
    #evaluate_model(data)

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


def get_trip_from_cluster(predicted_class, training_data, k=0):
    """
    Get an input from the training data with the predicted class as its target.

    Parameters:
    - predicted_theme (str): The predicted theme based on the user preferences.
    - training_data (pd.DataFrame): The DataFrame containing the training data.
    - k (int): The index of the road trip within the predicted theme class. Defaults to 0.

    Returns:
    - pd.Series: An input from the class with the predicted theme.

    If no more road trips are available in the cluster, an empty Series is returned.
    """
    # Filter training data based on the predicted theme
    cluster_data = training_data[training_data['utility'] == predicted_class]

    if k >= cluster_data.shape[0]:  # no more road trips in this cluster
        return pd.Series()
    else:
        trip_from_cluster = cluster_data.iloc[k]
        return trip_from_cluster.iloc[1:]


def prediction_output(predicted_class, data):
    """
    Print feature scores of recommended road trips for a predicted theme.

    Parameters:
    - predicted_class (str): The predicted theme based on user preferences.
    - data (pd.DataFrame): The DataFrame containing the training data.

    Prints:
    - Feature scores for a recommended road trip, grouped by theme.

    If there are no more road trip recommendations for the predicted theme, a message is printed,
    and the function terminates. Users can choose to continue receiving recommendations or exit.

    """
    print("We recommend a road trip with the following feature scores:")
    k = 0
    while True:
        road_trip = get_trip_from_cluster(predicted_class, data, k)
        if road_trip.empty:
            print('No more road trip recommendations')
            break
        for theme, pred in zip(theme_names, road_trip):
            print(f'{theme}: {pred}')
        if (input("Do you want another road trip (y/n)") != 'y'):
            break
        k = k + 1


data = pd.read_csv('data_files/clustered_data.csv')
user_pref = get_user_input()
model = train_network(data)
predicted_class = classify_user_preferences(model, user_pref)
prediction_output(predicted_class, data)
