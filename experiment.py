import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from your_custom_module import optimize_cnn_with_HEA_VLGA, optimize_cnn_with_grid_search, optimize_cnn_with_random_search

# Load and preprocess your dataset
X, y = load_and_preprocess_dataset()  # Replace with your actual dataset loading and preprocessing

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of runs for each optimization method
num_runs = 5  # You can adjust this number based on your experimental setup

# Initialize lists to store evaluation results
accuracy_results = []
precision_results = []
recall_results = []
f1_score_results = []

# Experiment 1: HEA-VLGA
for _ in range(num_runs):
    # Optimize CNN hyperparameters using HEA-VLGA
    best_cnn_model_HEA_VLGA = optimize_cnn_with_HEA_VLGA(X_train, y_train)
    
    # Evaluate the optimized model on the test set
    y_pred_HEA_VLGA = best_cnn_model_HEA_VLGA.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy_HEA_VLGA = accuracy_score(y_test, y_pred_HEA_VLGA)
    precision_HEA_VLGA = precision_score(y_test, y_pred_HEA_VLGA, average='weighted')
    recall_HEA_VLGA = recall_score(y_test, y_pred_HEA_VLGA, average='weighted')
    f1_score_HEA_VLGA = f1_score(y_test, y_pred_HEA_VLGA, average='weighted')
    
    # Append results to the lists
    accuracy_results.append(accuracy_HEA_VLGA)
    precision_results.append(precision_HEA_VLGA)
    recall_results.append(recall_HEA_VLGA)
    f1_score_results.append(f1_score_HEA_VLGA)

# Experiment 2: Grid Search
for _ in range(num_runs):
    # Optimize CNN hyperparameters using grid search
    best_cnn_model_grid_search = optimize_cnn_with_grid_search(X_train, y_train)
    
    # Evaluate the optimized model on the test set
    y_pred_grid_search = best_cnn_model_grid_search.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy_grid_search = accuracy_score(y_test, y_pred_grid_search)
    precision_grid_search = precision_score(y_test, y_pred_grid_search, average='weighted')
    recall_grid_search = recall_score(y_test, y_pred_grid_search, average='weighted')
    f1_score_grid_search = f1_score(y_test, y_pred_grid_search, average='weighted')
    
    # Append results to the lists
    accuracy_results.append(accuracy_grid_search)
    precision_results.append(precision_grid_search)
    recall_results.append(recall_grid_search)
    f1_score_results.append(f1_score_grid_search)

# Experiment 3: Random Search
for _ in range(num_runs):
    # Optimize CNN hyperparameters using random search
    best_cnn_model_random_search = optimize_cnn_with_random_search(X_train, y_train)
    
    # Evaluate the optimized model on the test set
    y_pred_random_search = best_cnn_model_random_search.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy_random_search = accuracy_score(y_test, y_pred_random_search)
    precision_random_search = precision_score(y_test, y_pred_random_search, average='weighted')
    recall_random_search = recall_score(y_test, y_pred_random_search, average='weighted')
    f1_score_random_search = f1_score(y_test, y_pred_random_search, average='weighted')
    
    # Append results to the lists
    accuracy_results.append(accuracy_random_search)
    precision_results.append(precision_random_search)
    recall_results.append(recall_random_search)
    f1_score_results.append(f1_score_random_search)

# Calculate mean and standard deviation of evaluation metrics
mean_accuracy = np.mean(accuracy_results)
std_accuracy = np.std(accuracy_results)

mean_precision = np.mean(precision_results)
std_precision = np.std(precision_results)

mean_recall = np.mean(recall_results)
std_recall = np.std(recall_results)

mean_f1_score = np.mean(f1_score_results)
std_f1_score = np.std(f1_score_results)

# Print and report the experimental results
print("Experimental Results:")
print(f"Mean Accuracy: {mean_accuracy:.4f} (Std Dev: {std_accuracy:.4f})")
print(f"Mean Precision: {mean_precision:.4f} (Std Dev: {std_precision:.4f})")
print(f"Mean Recall: {mean_recall:.4f} (Std Dev: {std_recall:.4f})")
print(f"Mean F1-Score: {mean_f1_score:.4f} (Std Dev: {std_f1_score:.4f})")

# You can also perform statistical tests (e.g., t-tests) to compare the methods statistically.
