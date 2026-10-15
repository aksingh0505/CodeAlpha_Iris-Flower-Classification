
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load the dataset
df = pd.read_csv("Iris.csv")

# 2. Separate features and target
X = df[
    ["SepalLengthCm", "SepalWidthCm",
     "PetalLengthCm", "PetalWidthCm"]
]

y = df["Species"].str.replace("Iris-", "", regex=False)

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Build the machine learning model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict flower species
y_pred = model.predict(X_test)

# 7. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred
)
plt.title("Iris Classification Confusion Matrix")
plt.show()

# 9. Predict a new flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(new_flower)

print("Predicted Species:", prediction[0])
