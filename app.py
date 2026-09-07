from fastapi import FastAPI
import pickle

app = FastAPI()

with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.post("/predict")
def predict(data: dict):

    values = data["data"]

    prediction = model.predict([values])

    return {"prediction": int(prediction[0])}