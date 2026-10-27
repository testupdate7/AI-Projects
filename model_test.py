
import joblib
from sklearn import datasets

# Load saved model
model = joblib.load("flower_model.pkl")

# Load flower names
iris = datasets.load_iris()

# Get measurements from user
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

# Prepare input
flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

# Predict species
prediction = model.predict(flower)

print("Predicted flower:", iris.target_names[prediction[0]])
