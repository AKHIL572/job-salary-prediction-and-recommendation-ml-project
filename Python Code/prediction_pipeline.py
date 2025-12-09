import joblib
import pandas as pd

# Load trained Gradient Boosting model & final columns
model = joblib.load("job_salary_prediction_model.pkl")
model_columns = joblib.load("model_columns.pkl")


def preprocess_input(user_input):

    df = pd.DataFrame([user_input])

    # ---------------------------
    # Derived feature engineering
    # ---------------------------
    df["range_experience"] = df["max_exp"] - df["min_exp"]
    df["avg_exp"] = (df["min_exp"] + df["max_exp"]) / 2
    df["salary_range"] = df["max_salary"] - df["min_salary"]
    df["avg_salary"] = (df["min_salary"] + df["max_salary"]) / 2

    df["skill_count"] = df["Key Skills"].apply(
        lambda x: len(str(x).split(",")) if isinstance(x, str) else 0
    )

    # -----------------------------------
    # One-hot encode SAME categorical cols
    # -----------------------------------
    categorical_cols = ['Industry', 'Functional Area', 'city',
                        'Role Category', 'seniority']

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --------------------------------------------------------
    # Reindex so final columns match exactly the trained model
    # Missing columns → filled with 0
    # --------------------------------------------------------
    df = df.reindex(columns=model_columns, fill_value=0)

    return df


def predict_salary(user_input):
    df_processed = preprocess_input(user_input)
    pred = model.predict(df_processed)[0]
    return pred


if __name__ == "__main__":

    sample_input = {
        "min_exp": 2,
        "max_exp": 5,
        "min_salary": 300000,
        "max_salary": 600000,
        "Industry": "IT",
        "Functional Area": "Engineering",
        "city": "Bangalore",
        "Role Category": "Software Developer",
        "seniority": "mid",
        "Key Skills": "Python,SQL,Machine Learning",
        "is_remote": 0
    }

    prediction = predict_salary(sample_input)
    print("Predicted Salary:", prediction)
