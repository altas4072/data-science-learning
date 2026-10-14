import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Student dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 30, 40, 50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

# Display dataset
print("STUDENT MARKS DATA")
print(df)

# Train Machine Learning Model
X = df[["Study_Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

# Predict marks
hours = float(input("\nEnter study hours: "))
prediction = model.predict(pd.DataFrame({"Study_Hours": [hours]}))

print("Predicted Marks:", round(prediction[0], 2))

# Plot graph
plt.scatter(df["Study_Hours"], df["Marks"])
plt.plot(df["Study_Hours"], model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()
