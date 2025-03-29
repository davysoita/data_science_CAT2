import pandas as pd
data = pd.read_csv("student_data.csv")  # Load student dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Handle missing values
data.fillna(data.median(), inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
data["program_choice"] = encoder.fit_transform(data["program_choice"])

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ["GPA", "SAT_score", "attendance_rate"]
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Split into training and test sets
X = data.drop(columns=["enrolled"])  # Features
y = data["enrolled"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


feature_importances = model.feature_importances_
plt.barh(X.columns, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Key Factors Influencing Enrollment")
plt.show()
