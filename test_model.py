import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

#  Import BEST model (choose based on comparison)
from sklearn.ensemble import GradientBoostingClassifier
# (If your result showed Random Forest best, replace above with RandomForestClassifier)

# Load dataset
data = pd.read_csv("upgraded_career_dataset.csv")

# Features and label
X = data.drop("career", axis=1)
y = data["career"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train mode
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Final Model Accuracy:", round(accuracy, 3))

# Save model
pickle.dump(model, open("career_model.pkl", "wb"))

print("Final model trained and saved successfully!")

