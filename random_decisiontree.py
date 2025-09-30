import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Classes and Functions for Decision Tree ---

class Node:
    """
    Represents a single node in the decision tree.
    It can be either an internal node (decision node) or a leaf node (prediction).
    """
    def __init__(self, feature_index=None, threshold=None, left_child=None, right_child=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

class DecisionTree:
    """
    Implements a Decision Tree Classifier from scratch using Entropy and Information Gain.
    """
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}

    def _calculate_entropy(self, y):
        """
        Calculates the Entropy for a set of labels.
        """
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        if total_samples == 0:
            return 0
        entropy = 0.0
        for count in counts:
            proportion = count / total_samples
            if proportion > 0:
                entropy -= proportion * np.log2(proportion)
        return entropy

    def _best_split(self, X, y):
        """
        Finds the best feature and threshold to split the data based on Information Gain.
        This is the core optimization step for a trained tree.
        """
        best_info_gain = -1.0
        best_feature_index = None
        best_threshold = None
        n_features = X.shape[1]
        parent_entropy = self._calculate_entropy(y)

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]
                entropy_left = self._calculate_entropy(y_left)
                entropy_right = self._calculate_entropy(y_right)

                n_left, n_right = len(y_left), len(y_right)
                total_samples = n_left + n_right
                weighted_avg_entropy = (n_left / total_samples) * entropy_left + (n_right / total_samples) * entropy_right
                information_gain = parent_entropy - weighted_avg_entropy

                if information_gain > best_info_gain:
                    best_info_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        if best_info_gain <= 0:
            return None, None
        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return Node(value=y[0])
        if len(y) < self.min_samples_split:
            return Node(value=np.bincount(y).argmax())
        if depth >= self.max_depth:
            return Node(value=np.bincount(y).argmax())

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return Node(value=np.bincount(y).argmax())

        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return Node(value=np.bincount(y).argmax())

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=feature_index, threshold=threshold,
                    left_child=left_child, right_child=right_child)

    def fit(self, X, y):
        unique_labels = np.unique(y)
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {i: label for i, label in enumerate(unique_labels)}
        y_encoded = np.array([self.label_encoder[label] for label in y])
        self.root = self._build_tree(X, y_encoded)

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left_child)
        else:
            return self._predict_single(x, node.right_child)

    def predict(self, X):
        predictions_encoded = [self._predict_single(x, self.root) for x in X]
        predictions = [self.reverse_label_encoder[pred_encoded] for pred_encoded in predictions_encoded]
        return np.array(predictions)

# --- NEW: Random Decision Tree Class ---
class RandomDecisionTree(DecisionTree):
    """
    A decision tree that builds itself by making random splits,
    simulating a randomly sampled function from the model family.
    """
    def _best_split(self, X, y):
        """
        Overrides the parent method to randomly select a feature and threshold.
        This demonstrates a non-optimized function from the model family.
        """
        n_features = X.shape[1]
        
        # Randomly choose a feature index to split on
        feature_index = np.random.choice(range(n_features))
        
        # Get unique thresholds for the chosen feature
        thresholds = np.unique(X[:, feature_index])
        
        if len(thresholds) == 0:
            return None, None
            
        # Randomly choose a threshold from the unique thresholds
        threshold = np.random.choice(thresholds)

        # Check if the split is valid (avoids empty left/right nodes)
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]

        if len(left_indices) == 0 or len(right_indices) == 0:
            # If the random choice results in an invalid split, return None
            # This is not a good practice for a real model, but demonstrates the
            # randomness of a non-optimized function.
            return None, None
            
        return feature_index, threshold

# --- Main execution block ---
csv_file_path = 'dataset/dataset.csv'

print("--- Comparing Trained vs. Random Decision Trees ---")

try:
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()
    symptom_columns = [col for col in df.columns if col.startswith('Symptom_')]
    df.dropna(subset=['Disease'], inplace=True)
    
    for col in symptom_columns:
        df[col] = df[col].fillna('nomoresymptoms')

    y = df['Disease'].values
    symptom_df = df[symptom_columns]
    all_symptoms_raw = pd.unique(symptom_df.values.ravel('K'))
    all_symptoms = sorted([s.strip() for s in all_symptoms_raw if isinstance(s, str) and s.strip()])

    X_one_hot = np.zeros((len(df), len(all_symptoms)), dtype=int)
    for i, row_index in enumerate(df.index):
        patient_symptoms = set(df.loc[row_index, symptom_columns].dropna().apply(lambda x: x.strip()))
        for j, symptom_name in enumerate(all_symptoms):
            if symptom_name in patient_symptoms:
                X_one_hot[i, j] = 1
    X = X_one_hot
    
    test_size = 0.3
    random_state = 42
    num_samples = X.shape[0]
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(num_samples)
    test_set_size = int(num_samples * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # --- Part 1: Train and evaluate the single, optimized tree ---
    optimized_tree = DecisionTree(max_depth=10)
    optimized_tree.fit(X_train, y_train)
    optimized_predictions = optimized_tree.predict(X_test)
    optimized_accuracy = accuracy_score(y_test, optimized_predictions)
    
    print(f"\nAccuracy of a Single, OPTIMIZED Tree: {optimized_accuracy:.4f}")
    
    # --- Part 2: Train and evaluate 100 random trees ---
    num_random_trees = 100
    random_accuracies = []
    
    print(f"\nTraining and Evaluating {num_random_trees} Random Trees...")
    for i in range(num_random_trees):
        # We set a new random seed for each tree to ensure they are unique
        np.random.seed(i)
        
        # Instantiate a random tree
        random_tree = RandomDecisionTree(max_depth=10)
        
        # Fit the tree (builds it with random splits)
        random_tree.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy
        random_predictions = random_tree.predict(X_test)
        random_accuracy = accuracy_score(y_test, random_predictions)
        random_accuracies.append(random_accuracy)
        
    avg_random_accuracy = np.mean(random_accuracies)
    max_random_accuracy = np.max(random_accuracies)
    min_random_accuracy = np.min(random_accuracies)
    
    print(f"Average Accuracy of {num_random_trees} Random Trees: {avg_random_accuracy:.4f}")
    print(f"Max Accuracy of Random Trees: {max_random_accuracy:.4f}")
    print(f"Min Accuracy of Random Trees: {min_random_accuracy:.4f}")
    
    # --- Part 3: Reflect and Compare ---
    print("\n--- Summary of Results ---")
    print("Comparison between a single optimized function and 100 random functions:")
    print(f"Optimized Tree Accuracy: {optimized_accuracy:.4f}")
    print(f"Average Random Tree Accuracy: {avg_random_accuracy:.4f}")
    print(f"Difference: {optimized_accuracy - avg_random_accuracy:.4f}")
    
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except KeyError as e:
    print(f"KeyError: Please ensure the target column 'Disease' and symptom columns exist in your CSV file. Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")