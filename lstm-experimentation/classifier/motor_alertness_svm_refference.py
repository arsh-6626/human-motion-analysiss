#!/usr/bin/env python
"""
SVMTrainer: A class to encapsulate SVM training on extracted sequence features.

This script performs the following steps:
    1. Loads features and labels from .npy files.
    2. Encodes string labels into numeric values.
    3. Splits the data into training and testing sets.
    4. Standardizes the features.
    5. Uses GridSearchCV to perform hyperparameter tuning for the SVM.
    6. Evaluates the best model on the test data.
    7. Saves the trained SVM model, the scaler, and a confusion matrix plot.
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

class SVMTrainer:
    def __init__(self, features_path, labels_path, test_size=0.2):
        """
        Initialize the SVMTrainer with file paths and test split size.
        
        Parameters:
            features_path (str): Path to the .npy file containing features.
            labels_path (str): Path to the .npy file containing labels.
            test_size (float): Fraction of data to reserve for testing.
        """
        self.features_path = features_path
        self.labels_path = labels_path
        self.test_size = test_size
        
        # Variables to store the data and objects used during training.
        self.X = None              # Feature array
        self.y = None              # Label array
        self.X_train = None        # Training features
        self.X_test = None         # Test features
        self.y_train = None        # Training labels
        self.y_test = None         # Test labels
        self.label_encoder = LabelEncoder()  # Encoder to convert string labels to numbers
        self.scaler = StandardScaler()       # Scaler for feature normalization
        self.model = None          # To hold the trained SVM model
        self.conf_mat_fig = None   # To hold the confusion matrix figure

    def load_data(self):
        """
        Load features and labels from .npy files.
        """
        self.X = np.load(self.features_path)
        self.y = np.load(self.labels_path)
        print(f"Loaded features from '{self.features_path}' with shape {self.X.shape}")
        print(f"Loaded labels from '{self.labels_path}' with shape {self.y.shape}")

    def preprocess_data(self):
        """
        Encode labels and split the data into training and test sets, then scale the features.
        """
        # Encode the labels (e.g., 'absent', 'twitching', 'walking') into numeric values.
        self.y = self.label_encoder.fit_transform(self.y)
        print("Encoded labels:", self.label_encoder.classes_)
        
        # Split the data into training and testing sets.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42, stratify=self.y
        )
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Test set size: {self.X_test.shape[0]} samples")
        
        # Standardize features by removing the mean and scaling to unit variance.
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Features have been standardized.")

    def train(self):
        """
        Train an SVM classifier using GridSearchCV for hyperparameter tuning.
        """
        # Initialize the SVM classifier.
        svm = SVC(probability=True)  # Enable probability estimates if needed.
        
        # Define a grid of hyperparameters to search over.
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        print("Starting Grid Search with parameters:", param_grid)
        
        # Create a GridSearchCV object to search for the best hyperparameters using 5-fold cross-validation.
        grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=2, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        print("Grid Search completed.")
        
        # Save the best estimator (the best performing SVM model).
        self.model = grid_search.best_estimator_
        print("Best parameters found:", grid_search.best_params_)

    def evaluate(self):
        """
        Evaluate the trained SVM model on the test set and generate a confusion matrix plot.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Use the model to predict labels on the test set.
        y_pred = self.model.predict(self.X_test)
        
        # Calculate the accuracy of the model.
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Test Accuracy: {:.2f}%".format(accuracy * 100))
        
        # Generate and print a detailed classification report.
        report = classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_)
        print("\nClassification Report:\n", report)
        
        # Compute the confusion matrix.
        conf_mat = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:\n", conf_mat)
        
        # Plot the confusion matrix as a heatmap.
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        # Save the current figure for later saving.
        self.conf_mat_fig = plt.gcf()
        plt.close()  # Close the plot so it doesn't display immediately
        
        return accuracy, report, conf_mat

    def save_artifacts(self, model_output, scaler_output, confusion_matrix_output):
        """
        Save the trained SVM model, the scaler, and the confusion matrix plot to disk.
        
        Parameters:
            model_output (str): File path for saving the trained SVM model.
            scaler_output (str): File path for saving the scaler.
            confusion_matrix_output (str): File path for saving the confusion matrix plot.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        joblib.dump(self.model, model_output)
        print("Trained SVM model saved to:", model_output)
        joblib.dump(self.scaler, scaler_output)
        print("Scaler saved to:", scaler_output)
        self.conf_mat_fig.savefig(confusion_matrix_output)
        print("Confusion matrix plot saved to:", confusion_matrix_output)

def main():
    parser = argparse.ArgumentParser(
        description="Train an SVM classifier on extracted sequence features using OOP."
    )
    parser.add_argument('--features_path', type=str, default='all_features.npy',
                        help='Path to the .npy file containing the extracted features.')
    parser.add_argument('--labels_path', type=str, default='all_labels.npy',
                        help='Path to the .npy file containing the labels.')
    parser.add_argument('--model_output', type=str, default='svm_model.joblib',
                        help='Path to save the trained SVM model.')
    parser.add_argument('--scaler_output', type=str, default='scaler.joblib',
                        help='Path to save the scaler used for normalization.')
    parser.add_argument('--confusion_matrix_output', type=str, default='confusion_matrix.png',
                        help='Path to save the confusion matrix plot.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of the dataset to reserve for testing (default is 0.2).')
    args = parser.parse_args()
    trainer = SVMTrainer(args.features_path, args.labels_path, args.test_size)
    trainer.load_data()
    trainer.preprocess_data()
    trainer.train()
    trainer.evaluate()
    trainer.save_artifacts(args.model_output, args.scaler_output, args.confusion_matrix_output)

if __name__ == '__main__':
    main()


