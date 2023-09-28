import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold  # Use StratifiedKFold for classification tasks
from sklearn.metrics import accuracy_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Define the number of cross-validation folds
num_folds = 5  # You can adjust this number as needed

# Define a function to create and compile a CNN model based on hyperparameters
def create_cnn_model(learning_rate, batch_size, num_filters, num_layers, dropout_rate):
    model = Sequential()
    model.add(Conv2D(num_filters, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    for _ in range(num_layers - 1):
        model.add(Conv2D(num_filters, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# Create a function to evaluate the fitness of a CNN model using cross-validation
def evaluate_fitness(cnn_model, X, y):
    # Convert labels to one-hot encoding
    # Assuming you have 10 classes (0-9 digits)
    num_classes = 10
    y_one_hot = keras.utils.to_categorical(y, num_classes)
    
    # Initialize cross-validation
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Initialize a list to store accuracy scores for each fold
    accuracy_scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in kfold.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_one_hot[train_idx], y_one_hot[val_idx]
        
        # Create and compile the CNN model
        model = create_cnn_model(
            cnn_model['learning_rate'],
            cnn_model['batch_size'],
            cnn_model['num_filters'],
            cnn_model['num_layers'],
            cnn_model['dropout_rate']
        )
        
        # Train the model on the training data
        model.fit(X_train, y_train, batch_size=cnn_model['batch_size'], epochs=10, verbose=0)
        
        # Evaluate the model on the validation data
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Append the accuracy to the list
        accuracy_scores.append(accuracy)
    
    # Calculate the mean accuracy across folds
    mean_accuracy = np.mean(accuracy_scores)
    
    return mean_accuracy

# Assuming X and y are your dataset and labels
# Initialize the population of CNN models (you can use the previous code for this)
population = [...]

# Evaluate the fitness of each CNN model in the population
for i, cnn_model in enumerate(population):
    fitness_score = evaluate_fitness(cnn_model, X, y)
    print(f"Model {i + 1} - Fitness Score: {fitness_score:.4f}")
