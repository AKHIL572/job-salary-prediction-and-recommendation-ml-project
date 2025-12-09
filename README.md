# 💼 AI Job Salary Prediction & Intelligent Job Recommendation System
### 📘 Overview

This project is an end-to-end Machine Learning + NLP-based Job Analysis System that contains:

✅ 1. Job Salary Prediction (ML Regression Model)

Predicts the expected average salary of a job based on experience, industry, skills, job role, location, and more.

✅ 2. Intelligent Job Recommendation System (NLP + Cosine Similarity)

Recommends similar jobs based on job title, skills, role, functional area, and industry using TF-IDF and Cosine Similarity.

This system follows the complete Data Science workflow—data cleaning, feature engineering, ML modeling, text vectorization, and deployment via Streamlit.

### 📁 Project Structure
#### Section	Description
1️⃣ Data Cleaning & Preprocessing	Cleaned all raw job postings: processed salary ranges, extracted experience, standardized text fields, removed duplicates, handled missing values.
2️⃣ Exploratory Data Analysis (EDA)	Histograms, boxplots, correlations, outlier handling, salary distributions by industry/location.
3️⃣ Feature Engineering	Created new features like avg_salary, range_experience, skill_count, seniority, is_remote, city extraction, and one-hot encoding.
4️⃣ Salary Model Training	Trained multiple ML models (LR, DT, RF, GB). Tuned hyperparameters and selected the best model (Gradient Boosting Regressor).
5️⃣ Recommendation System (NLP)	Merged text features → TF-IDF vectorization → cosine similarity matrix → top job recommendations.
6️⃣ Deployment-Ready Artifacts	Exported job_salary_prediction_model.pkl, model_columns.pkl, tfidf_vectorizer.pkl, job_similarity_matrix.pkl, and dataset.
7️⃣ Streamlit Web App	Integrated both systems into a dual-tab interface: Salary Predictor + Job Recommender.
### 📦 Dataset Files
#### File	Description
marketing_sample_for_naukri…csv	Original dataset downloaded.
cleaned_final_dataset.csv	Fully cleaned and processed dataset after feature engineering.
recommendation_data.pkl	Data used for job recommendation (with combined text).
job_similarity_matrix.pkl	Precomputed cosine similarity matrix for recommendations.
tfidf_vectorizer.pkl	Saved TF-IDF vectorizer.
job_salary_prediction_model.pkl	Final trained Gradient Boosting model for salary prediction.
model_columns.pkl	Final feature column order used during model training.
### 📊 Model Performance (Salary Prediction)
#### Metric	Value
RMSE	(insert value from notebook)
MAE	(insert value)
R² Score	(insert value)
✅ Best Model: Gradient Boosting Regressor

Delivered the lowest RMSE and best predictive accuracy.

### 🧠 Job Recommendation System
#### ✔ Combined Features Used

Job Title

Key Skills

Functional Area

Industry

Role

City

#### ✔ Technique Used

TF-IDF vectorization (5000 features)

Cosine similarity for job matching

Retrieves top 10 most similar jobs

### 🧮 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

NLTK / TF-IDF

Cosine Similarity

Streamlit

Joblib (model saving)

### 🎯 Key Learnings

Cleaning and preprocessing large job datasets

Feature extraction from salary text

NLP methods (TF-IDF + Cosine Similarity)

Regression modeling and hyperparameter tuning

Deploying ML models with Streamlit

Creating multi-page/dual-tab ML applications

### 🚀 How to Use the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the application
streamlit run app.py

3️⃣ The web app gives you:

Tab 1 → Salary Prediction

Tab 2 → Job Recommendations

### 📎 Appendix

This repository contains:

Jupyter Notebook source code

Cleaned datasets

TF-IDF and similarity matrices

Trained ML salary prediction model

Streamlit deployment files

Visualizations and EDA plots
