import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def load_and_preprocess_data(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Separate features and target
    X = df.drop('phase', axis=1)
    y = df['phase']
    
    return X, y

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf, X_test, y_test

def evaluate_model(clf, X_test, y_test):
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Print the classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

def save_model(clf, filename):
    # Save the trained model
    joblib.dump(clf, filename)
    print(f"Model saved as {filename}")

def main():
    # List of exercise types
    exercises = ["pushup", "plank", "star_jumps", "squats"]
    
    for exercise in exercises:
        print(f"\nTraining model for {exercise}...")
        csv_file = f'Data_and_models/CSV/{exercise}_coordinates.csv'
        
        # Load and preprocess data
        X, y = load_and_preprocess_data(csv_file)
        
        # Train the model
        clf, X_test, y_test = train_model(X, y)
        
        # Evaluate the model
        evaluate_model(clf, X_test, y_test)
        
        # Save the model
        save_model(clf, f'Data_and_models/{exercise}_model.joblib')

if __name__ == "__main__":
    main()
