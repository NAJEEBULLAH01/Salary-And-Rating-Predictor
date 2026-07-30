import streamlit as st
import pandas as pd
import pickle
import zipfile
import os
import time

st.set_page_config(page_title="Pro Salary Predictor", layout="wide")

# Pro CSS Animations - no external file
st.markdown("""
<style>
@keyframes fadeInUp {from{opacity:0; transform:translateY(20px);} to{opacity:1; transform:translateY(0);}}
.block-container {animation: fadeInUp 0.5s ease;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {border-radius: 12px; padding: 10px 18px; transition: all 0.3s;}
.stTabs [aria-selected="true"] {background: linear-gradient(90deg,#2563eb,#3b82f6); color:white!important;}
.metric-card {background: linear-gradient(135deg,#2563eb,#3b82f6); color:white; padding:2.5rem; border-radius:20px;
              text-align:center; animation: fadeInUp 0.6s ease; box-shadow:0 10px 30px rgba(37,99,235,0.35);}
.metric-card h1 {font-size: 2.8rem; margin: 0;}
.stButton>button {background: linear-gradient(90deg,#2563eb,#3b82f6); color:white; border:none; border-radius:12px;
                  padding:0.9rem; font-weight:700; font-size:1.1rem; width:100%; transition:0.2s;}
.stButton>button:hover {transform: translateY(-3px); box-shadow:0 8px 20px rgba(37,99,235,0.4);}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    MODEL_ZIP = "SalaryPrediction1"
    MODEL_PKL = "SalaryPrediction1.pickle"
    if not os.path.exists(MODEL_PKL):
        if os.path.exists(MODEL_ZIP):
            with st.spinner("🚀 Loading model..."):
                with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
                    zip_ref.extractall(".")
    return pickle.load(open(MODEL_PKL, "rb"))

model = load_model()

st.title("💼 Pro Salary & Rating Predictor")
st.caption("Predict Avg Salary and Company Rating with AI")

tab1, tab2, tab3 = st.tabs(["👨‍💼 Job Details", "🏢 Company Info", "💰 Salary & Predict"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("age", 18, 65, 28)
        job_title = st.text_input("Job Title", "Data Scientist")
        job_state = st.text_input("job_state")
        skills = st.multiselect("Tech Skills", ['python_yn', 'excel', 'spark', 'aws', 'R_yn'])
        hourly = st.radio("hourly", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
    with col2:
        job_description = st.text_area("Job Description", height=150)
        employer_provided = st.radio("employer_provided", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        company_name = st.text_input("Company Name")
        headquarters = st.text_input("Headquarters")
        location = st.text_input("Location")
        sector = st.text_input("Sector")
        industry = st.text_input("Industry")
    with col4:
        founded = st.number_input("Founded", 1800, 2026, 2000)
        size = st.text_input("Size")
        ownership = st.text_input("Type of ownership")
        revenue = st.text_input("Revenue")
        competitors = st.text_area("Competitors")
        company_txt = st.text_area("company_txt", height=100)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        min_salary = st.number_input("min_salary", 0, 1000000, 0, step=5000)
        max_salary = st.number_input("max_salary", 0, 1000000, 0, step=5000)
    with col6:
        st.write("")
        st.write("")
        submit = st.button("🚀 Predict Now", use_container_width=True)

if submit:
    with st.spinner("🤖 AI is predicting..."):
        time.sleep(0.8)

        # Convert skills to 0/1
        skill_dict = {s: 1 if s in skills else 0 for s in ['python_yn', 'excel', 'spark', 'aws', 'R_yn']}

        # EXACT ORDER as your columns
        input_dict = {
            'excel': skill_dict['excel'],
            'spark': skill_dict['spark'],
            'R_yn': skill_dict['R_yn'],
            'python_yn': skill_dict['python_yn'],
            'age': age,
            'same_state': 0, # you didn't add input for this, default 0. Change if needed
            'Founded': founded,
            'max_salary': max_salary,
            'min_salary': min_salary,
            'aws': skill_dict['aws'],
            'hourly': hourly,
            'employer_provided': employer_provided,
            'job_state': job_state,
            'company_txt': company_txt,
            'Job Title': job_title,
            'Sector': sector,
            'Industry': industry,
            'Type of ownership': ownership,
            'Size': size,
            'Headquarters': headquarters,
            'Location': location,
            'Company Name': company_name,
            'Job Description': job_description,
            'Revenue': revenue,
            'Competitors': competitors
        }

        input_df = pd.DataFrame([input_dict])
        pred = model.predict(input_df)[0] # pred[0]=avg_salary, pred[1]=Rating

        pred[1] = min(5.0, max(1.0, pred[1])) # clamp Rating 1-5

    st.markdown("### 🎯 Prediction Results")
    res1, res2 = st.columns(2)
    with res1:
        st.markdown(f"<div class='metric-card'><h3>Avg Salary</h3><h1>{pred[0]:,.2f}</h1></div>", unsafe_allow_html=True)
    with res2:
        st.markdown(f"<div class='metric-card'><h3>Rating</h3><h1>{pred[1]:.1f} ⭐</h1></div>", unsafe_allow_html=True)
