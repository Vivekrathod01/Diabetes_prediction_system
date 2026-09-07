import streamlit as st
import requests

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("🩺 Diabetes Prediction System")
st.write("Enter all patient details.")

pregnancies = st.text_input("Pregnancies")
glucose = st.text_input("Glucose")
blood_pressure = st.text_input("Blood Pressure")
skin_thickness = st.text_input("Skin Thickness")
insulin = st.text_input("Insulin")
bmi = st.text_input("BMI")
pedigree = st.text_input("Diabetes Pedigree Function")
age = st.text_input("Age")


if st.button("🔍 Predict Diabetes", use_container_width=True):

    # Check empty fields
    if not all([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        pedigree,
        age
    ]):
        st.warning("Please fill in all fields.")

    else:

        try:
            data = [
                float(pregnancies),
                float(glucose),
                float(blood_pressure),
                float(skin_thickness),
                float(insulin),
                float(bmi),
                float(pedigree),
                float(age)
            ]

            response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"data": data}
)
            # Check what FastAPI actually returned
            if response.status_code != 200:
                st.error("FastAPI returned an error.")
                st.write(response.json())

            else:
                result = response.json()

                if "prediction" not in result:
                    st.error("Prediction was not returned by FastAPI.")
                    st.write(result)

                elif result["prediction"] == 1:
                    st.error("⚠️ Higher Risk of Diabetes")

                else:
                    st.success("✅ Lower Risk of Diabetes")

        except ValueError:
            st.warning("Please enter numbers only.")

        except requests.exceptions.ConnectionError:
            st.error("❌ FastAPI server is not running.")