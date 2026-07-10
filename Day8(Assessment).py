import streamlit as st
import pandas as pd
import pickle

# Load the trained model and encoders
with open("job_role_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.set_page_config(page_title="Job Role Predictor", page_icon="💼", layout="centered")

st.title("💼 Job Role Prediction System")
st.markdown("### Enter your skill levels to predict the best matching job role")

# List of skills
skill_list = [
    "Database Fundamentals", "Computer Architecture", "Distributed Computing Systems",
    "Cyber Security", "Networking", "Software Development", "Programming Skills",
    "Project Management", "Computer Forensics Fundamentals", "Technical Communication",
    "AI ML", "Software Engineering", "Business Analysis", "Communication skills",
    "Data Science", "Troubleshooting skills", "Graphics Designing"
]

options = ["Not Interested", "Poor", "Beginner", "Average", "Intermediate", "Excellent", "Professional"]

# Create input fields
skills_input = {}
col1, col2 = st.columns(2)

for i, skill in enumerate(skill_list):
    with (col1 if i % 2 == 0 else col2):
        skills_input[skill] = st.selectbox(f"**{skill}**", options, index=3)

# Prediction
if st.button("🔮 Predict My Best Job Role", type="primary", use_container_width=True):
    skill_map = {
        'Not Interested': 0, 'Poor': 1, 'Beginner': 2, 'Average': 3,
        'Intermediate': 4, 'Excellent': 5, 'Professional': 6
    }
    
    input_data = [[skill_map[skills_input[s]] for s in skill_list]]
    input_df = pd.DataFrame(input_data, columns=skill_list)
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    predicted_role = le.inverse_transform(prediction)[0]
    
    st.success(f"**🎉 Your Best Matching Role: {predicted_role}**")
    st.balloons()
    
    st.info("This prediction is based on your entered skill levels.")
