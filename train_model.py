import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load dataset
data = pd.read_csv(r"C:\Users\sanaj\Downloads\upgraded_career_dataset.csv")

# Features and label
X = data.drop("career", axis=1)
y = data["career"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

print(" Model Comparison Results:\n")

best_model = None
best_accuracy = 0

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{name} Accuracy: {round(acc, 3)}")
    
    # Track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

print("\n Best Model:", best_name)
print("Best Accuracy:", round(best_accuracy, 3))

# Save best model
import pickle
pickle.dump(best_model, open("career_model.pkl", "wb"))

print("\n Best model saved as career_model.pkl")


