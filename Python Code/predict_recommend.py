import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------
# Load all models and data
# ---------------------------------------
model = joblib.load("job_salary_prediction_model.pkl")
model_columns = joblib.load("model_columns.pkl")

df_rec = joblib.load("recommendation_data.pkl")
similarity_matrix = joblib.load("job_similarity_matrix.pkl")

# ---------------------------------------
# Page config
# ---------------------------------------
st.set_page_config(page_title="AI Job Salary + Recommendation System", layout="wide")

st.title("💼 AI Job Salary Prediction + 🔍 Intelligent Job Recommendation System")

# ---------------------------------------
# Helper function for dropdown extraction
# ---------------------------------------
def extract_categories(prefix):
    return sorted([col.replace(prefix, "") for col in model_columns if col.startswith(prefix)])

industry_list = extract_categories("Industry_")
functional_list = extract_categories("Functional Area_")
city_list = extract_categories("city_")
role_category_list = extract_categories("Role Category_")
seniority_list = extract_categories("seniority_")

industry_list.insert(0, "Select Industry")
functional_list.insert(0, "Select Functional Area")
city_list.insert(0, "Select City")
role_category_list.insert(0, "Select Role Category")

# Manual seniority choices
seniority_list = ["junior", "mid", "senior"]

# ---------------------------------------
# TABS: Salary Prediction + Recommendation
# ---------------------------------------
tab1, tab2 = st.tabs(["💰 Salary Prediction", "🔍 Job Recommendation"])

# =======================================================================================
# TAB 1 : SALARY PREDICTION
# =======================================================================================
with tab1:

    st.header("💰 Job Salary Prediction")

    industry = st.selectbox("Industry", industry_list)
    functional_area = st.selectbox("Functional Area", functional_list)
    city = st.selectbox("City", city_list)
    role_category = st.selectbox("Role Category", role_category_list)
    seniority = st.selectbox("Seniority (junior / mid / senior)", seniority_list)

    min_exp = st.number_input("Minimum Experience (years)", 0, 40, 1)
    max_exp = st.number_input("Maximum Experience (years)", 0, 40, 3)

    min_salary = st.number_input("Minimum Salary (if known)", min_value=0, value=300000)
    max_salary = st.number_input("Maximum Salary (if known)", min_value=0, value=600000)

    key_skills = st.text_area("Key Skills (comma-separated)", "Python, SQL, Machine Learning")

    is_remote = st.selectbox("Is Remote Job?", ["No", "Yes"])
    is_remote = 1 if is_remote == "Yes" else 0

    if st.button("Predict Salary"):

        # Derived features
        range_experience = max_exp - min_exp
        avg_exp = (min_exp + max_exp) / 2
        salary_range = max_salary - min_salary
        skill_count = len(key_skills.split(","))

        input_data = {
            "min_exp": [min_exp],
            "max_exp": [max_exp],
            "min_salary": [min_salary],
            "max_salary": [max_salary],
            "range_experience": [range_experience],
            "avg_exp": [avg_exp],
            "salary_range": [salary_range],
            "avg_salary": [(min_salary + max_salary) / 2],
            "skill_count": [skill_count],
            "is_remote": [is_remote],
            "Industry": [industry],
            "Functional Area": [functional_area],
            "city": [city],
            "Role Category": [role_category],
            "seniority": [seniority],
            "Key Skills": [key_skills]
        }

        df_input = pd.DataFrame(input_data)

        df_input = pd.get_dummies(df_input, columns=[
            "Industry", "Functional Area", "city", "Role Category", "seniority"
        ], drop_first=True)

        for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[model_columns]

        prediction = model.predict(df_input)[0]

        st.success(f"Predicted Salary: ₹ {prediction:,.2f}")


# =======================================================================================
# TAB 2 : JOB RECOMMENDATION SYSTEM
# =======================================================================================
with tab2:

    st.header("🔍 Intelligent Job Recommendation System")

    job_list = df_rec["Job Title"].dropna().unique()
    selected_job = st.selectbox("Select a Job Title", job_list)

    def recommend_jobs(job_index, top_n=10):
        similarity_scores = list(enumerate(similarity_matrix[job_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        top_jobs = similarity_scores[1:top_n+1]

        recommended = df_rec.iloc[[i[0] for i in top_jobs]][[
            "Job Title", "Key Skills", "Industry", "Functional Area", "Role", "city"
        ]]

        return recommended

    if st.button("Recommend Jobs"):
        job_index = df_rec[df_rec["Job Title"] == selected_job].index[0]
        results = recommend_jobs(job_index, top_n=10)
        st.subheader("Top Matching Jobs")
        st.dataframe(results)
