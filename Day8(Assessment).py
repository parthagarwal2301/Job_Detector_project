%%writefile Day8(Assessment).py

import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("random_forest_model.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.title("💼 Job Role Prediction System")
st.write("Select your skill level for each subject.")

Database_Fundamentals = st.selectbox("Database Fundamentals", ("Beginner", "Intermediate", "Professional"))
Computer_Architecture = st.selectbox("Computer Architecture", ("Beginner", "Intermediate", "Professional"))
Distributed_Computing_Systems = st.selectbox("Distributed Computing Systems", ("Beginner", "Intermediate", "Professional"))
Cyber_Security = st.selectbox("Cyber Security", ("Beginner", "Intermediate", "Professional"))
Networking = st.selectbox("Networking", ("Beginner", "Intermediate", "Professional"))
Software_Development = st.selectbox("Software Development", ("Beginner", "Intermediate", "Professional"))
Programming_Skills = st.selectbox("Programming Skills", ("Beginner", "Intermediate", "Professional"))
Project_Management = st.selectbox("Project Management", ("Beginner", "Intermediate", "Professional"))
Computer_Forensics = st.selectbox("Computer Forensics Fundamentals", ("Beginner", "Intermediate", "Professional"))
Technical_Communication = st.selectbox("Technical Communication", ("Beginner", "Intermediate", "Professional"))
AI_ML = st.selectbox("AI ML", ("Beginner", "Intermediate", "Professional"))
Software_Engineering = st.selectbox("Software Engineering", ("Beginner", "Intermediate", "Professional"))
Business_Analysis = st.selectbox("Business Analysis", ("Beginner", "Intermediate", "Professional"))
Communication_Skills = st.selectbox("Communication Skills", ("Beginner", "Intermediate", "Professional"))
Data_Science = st.selectbox("Data Science", ("Beginner", "Intermediate", "Professional"))
Troubleshooting_Skills = st.selectbox("Troubleshooting Skills", ("Beginner", "Intermediate", "Professional"))
Graphics_Designing = st.selectbox("Graphics Designing", ("Beginner", "Intermediate", "Professional"))

if st.button("Predict Job Role"):

    user_input = [
        encoders["Database Fundamentals"].transform([Database_Fundamentals])[0],
        encoders["Computer Architecture"].transform([Computer_Architecture])[0],
        encoders["Distributed Computing Systems"].transform([Distributed_Computing_Systems])[0],
        encoders["Cyber Security"].transform([Cyber_Security])[0],
        encoders["Networking"].transform([Networking])[0],
        encoders["Software Development"].transform([Software_Development])[0],
        encoders["Programming Skills"].transform([Programming_Skills])[0],
        encoders["Project Management"].transform([Project_Management])[0],
        encoders["Computer Forensics Fundamentals"].transform([Computer_Forensics])[0],
        encoders["Technical Communication"].transform([Technical_Communication])[0],
        encoders["AI ML"].transform([AI_ML])[0],
        encoders["Software Engineering"].transform([Software_Engineering])[0],
        encoders["Business Analysis"].transform([Business_Analysis])[0],
        encoders["Communication skills"].transform([Communication_Skills])[0],
        encoders["Data Science"].transform([Data_Science])[0],
        encoders["Troubleshooting skills"].transform([Troubleshooting_Skills])[0],
        encoders["Graphics Designing"].transform([Graphics_Designing])[0]
    ]

    prediction = model.predict([user_input])
    result = encoders["Role"].inverse_transform(prediction)

    st.success("Predicted Job Role: " + result[0])
