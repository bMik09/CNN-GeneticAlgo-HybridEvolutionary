import random

# Define the hyperparameter search space
learning_rate_range = [0.001, 0.01, 0.1]
batch_size_range = [32, 64, 128]
num_filters_range = [16, 32, 64]
num_layers_range = [2, 3, 4]
dropout_rate_range = [0.2, 0.3, 0.4]

# Define the population size
population_size = 27

# Initialize an empty list to store the population of CNN models
population = []

# Generate CNN models with variable-length encoding
for _ in range(population_size):
    # Define a random architecture by selecting values from the search space
    learning_rate = random.choice(learning_rate_range)
    batch_size = random.choice(batch_size_range)
    num_filters = random.choice(num_filters_range)
    num_layers = random.choice(num_layers_range)
    dropout_rate = random.choice(dropout_rate_range)
    
    # Create a CNN model with the selected hyperparameters
    cnn_model = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_filters': num_filters,
        'num_layers': num_layers,
        'dropout_rate': dropout_rate,
        # Add more hyperparameters as needed
    }
    
    # Append the CNN model to the population
    population.append(cnn_model)

# Print the initialized population
for i, cnn_model in enumerate(population):
    print(f"Model {i + 1}:")
    print(cnn_model)
    print("\n")
