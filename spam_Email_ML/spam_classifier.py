import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.validation import check_X_y

# Load and clean the dataset
def load_data(filename='spam.csv'):
    try:
        # Read tab-delimited CSV
        data = pd.read_csv(filename, encoding='latin-1', sep='\t', names=['label', 'message'])
        
        # Drop empty rows and incorrect labels
        data.dropna(subset=['label', 'message'], inplace=True)
        data['label'] = data['label'].str.strip().str.lower()
        print(f"Unique labels before mapping: {data['label'].unique()}")  # Debug info
        data['label'] = (data['label'] == 'spam').astype(int)  # Convert to binary: spam=1, ham=0
        print(f"Unique labels after mapping: {data['label'].unique()}")  # Debug info

        if data.empty:
            raise ValueError("Dataset is empty after cleaning!")

        # Check class distribution
        class_counts = data['label'].value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")  # Debugging info

        return data
    except FileNotFoundError:
        print("Error: The file was not found. Please check the filename and path.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

# Preprocess the data
def preprocess_data(data):
    data['message'] = data['message'].astype(str).str.strip()  # Ensure messages are strings
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    X = vectorizer.fit_transform(data['message']).toarray()
    y = data['label'].astype(int)

    # Validate data
    try:
        X, y = check_X_y(X, y)
    except ValueError as e:
        print(f"Data validation error: {e}")
        exit(1)

    print(f"Preprocessed data shapes - X: {X.shape}, y: {y.shape}")  # Debugging info

    return X, y, vectorizer

# Train and evaluate models
def train_and_evaluate(X, y):
    # Check for class balance
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        print(f"Error: Dataset contains only one class: {unique_classes}. At least two classes (0 and 1) are required.")
        exit(1)
    print(f"Class distribution: {dict(zip(unique_classes, counts))}")  # Debugging info

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=500, solver='liblinear'),  # Added solver
        "SGD Classifier": SGDClassifier(loss="log_loss", random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            print(f"\n{name} Accuracy: {acc:.4f}")
            print(classification_report(y_test, predictions))
            trained_models[name] = model
        except Exception as e:
            print(f"Error training {name}: {e}")

    return trained_models

def classify_email(email_text, models, vectorizer):
    """Classify a single email text using all trained models."""
    # Vectorize the input email
    email_vectorized = vectorizer.transform([email_text])
    
    # Get predictions from all models
    results = {}
    for name, model in models.items():
        try:
            prediction = model.predict(email_vectorized)[0]
            results[name] = "SPAM" if prediction == 1 else "HAM"
        except Exception as e:
            results[name] = f"Error: {e}"
    
    return results

def get_user_input():
    """Get email text from user input."""
    print("\nEnter the email text (press Enter twice to finish):\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)

# Main function
if __name__ == "__main__":
    # Train the models first
    print("Training models...")
    data = load_data()
    X, y, vectorizer = preprocess_data(data)
    models = train_and_evaluate(X, y)
    
    # Interactive classification loop
    while True:
        print("\n" + "="*50)
        print("Email Spam Classifier")
        print("="*50)
        print("1. Classify an email")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1 or 2): ")
        
        if choice == "2":
            print("\nGoodbye!")
            break
        elif choice == "1":
            email_text = get_user_input()
            if email_text.strip():
                print("\nClassifying...")
                results = classify_email(email_text, models, vectorizer)
                
                print("\nClassification Results:")
                print("-" * 30)
                for model_name, prediction in results.items():
                    print(f"{model_name}: {prediction}")
            else:
                print("\nNo email text provided!")
        else:
            print("\nInvalid choice! Please enter 1 or 2.")