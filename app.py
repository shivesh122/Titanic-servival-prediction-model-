import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

# Load model
MODEL_PATH = "titanic_best_model.pkl"
class SimpleTitanicModel:
    """A tiny heuristic model with scikit-learn-like API."""

    def __init__(self):
        pass

    def _score(self, X):
        import numpy as np
        import pandas as pd
        s = np.full(len(X), 0.2, dtype=float)
        sex = X['Sex'].astype(str).str.lower().fillna('male')
        pclass = pd.to_numeric(X['Pclass'], errors='coerce').fillna(3)
        farepp = pd.to_numeric(X['FarePerPerson'], errors='coerce').fillna(0.0)
        alone = pd.to_numeric(X['IsAlone'], errors='coerce').fillna(1)

        s += (sex == 'female') * 0.5
        s += (pclass == 1).astype(float) * 0.15
        s += (farepp > 20).astype(float) * 0.1
        s -= ((alone == 1) & (sex == 'male')).astype(float) * 0.15

        return np.clip(s, 0.01, 0.99)

    def predict_proba(self, X):
        p1 = self._score(X)
        p0 = 1 - p1
        import numpy as np
        return np.vstack([p0, p1]).T

    def predict(self, X):
        return (self._score(X) >= 0.5).astype(int)
        

model = load_model()

# Features used in model
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked',
            'Title','FamilySize','IsAlone','Deck','FarePerPerson']

# Feature engineering function
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Make sure required raw columns exist (best-effort fill if missing for single predictions)
    defaults = {
        'Name': 'Doe, Mr. John',
        'Cabin': None,
        'Ticket': '',
        'Embarked': df['Embarked'].mode()[0] if 'Embarked' in df and df['Embarked'].notna().any() else 'S',
        'Fare': df['Fare'].median() if 'Fare' in df and df['Fare'].notna().any() else 7.25,
        'Age': df['Age'].median() if 'Age' in df and df['Age'].notna().any() else 30,
        'SibSp': 0,
        'Parch': 0
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    df['Title'] = df['Name'].astype(str).str.extract(r',\s*([^.]*)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss').replace('Mme','Mrs')
    vc = df['Title'].value_counts()
    rare = vc[vc < 10].index
    df.loc[df['Title'].isin(rare), 'Title'] = 'Rare'

    df['FamilySize'] = df['SibSp'].fillna(0) + df['Parch'].fillna(0) + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Deck'] = df.get('Cabin', pd.Series([None]*len(df))).fillna('U').astype(str).str[0]
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Embarked' in df:
        if df['Embarked'].isna().any():
            df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0] if df['Embarked'].notna().any() else 'S')
    else:
        df['Embarked'] = 'S'
    df['FarePerPerson'] = df['Fare'] / df['FamilySize'].replace(0, 1)
    return df

# Title
st.title("🚢 Titanic Survival Prediction App")

if model is None:
    st.error("Model not found. Please place a pickle at models/titanic_best_model.pkl")
else:
    st.success("Model loaded. Ready to predict!")

st.markdown("---")

# Batch CSV upload
st.subheader("📄 Batch CSV Prediction")
with st.expander("CSV Template & Notes", expanded=False):
    template = StringIO()
    template.write("PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n")
    template.write('1,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S\n')
    template.seek(0)
    st.download_button("Download CSV Template", data=template.getvalue().encode('utf-8'), file_name="titanic_template.csv")
    st.caption("Upload a CSV with headers like the Kaggle Titanic dataset.")

uploaded_file = st.file_uploader("Upload CSV file with passengers", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df_fe = feature_engineer(df)
        missing = [c for c in features if c not in df_fe.columns]
        if missing:
            st.warning(f"Engineered features missing: {missing}. Please check your input.")
        X = df_fe.reindex(columns=features, fill_value=np.nan)
        preds = model.predict(X)
        df['PredictedSurvived'] = preds
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
                df['SurvivalProb'] = proba[:, 1]
        st.dataframe(df.head(20))
        st.download_button("Download Predictions CSV", df.to_csv(index=False).encode('utf-8'), "predictions.csv")
    except Exception as e:
        st.error(f"Error while predicting: {e}")

st.markdown("---")

# Single passenger prediction
st.subheader("🧍 Single Passenger Prediction")
with st.form("single_form"):
    name = st.text_input("Name", "Doe, Mr. John")
    pclass = st.selectbox("Pclass", [1,2,3], index=2)
    sex = st.selectbox("Sex", ["male","female"])
    age = st.number_input("Age", 0, 120, 30)
    sibsp = st.number_input("SibSp", 0, 10, 0)
    parch = st.number_input("Parch", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 10000.0, 7.25)
    embarked = st.selectbox("Embarked", ["C","Q","S"], index=2)
    cabin = st.text_input("Cabin (optional)", "")
    submitted = st.form_submit_button("Predict")

if submitted:
    single_df = pd.DataFrame([{
        'PassengerId': 0,
        'Pclass': pclass,
        'Name': name,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Ticket': '',
        'Fare': fare,
        'Cabin': cabin if cabin else None,
        'Embarked': embarked
    }])
    single_fe = feature_engineer(single_df)
    X1 = single_fe.reindex(columns=features, fill_value=np.nan)
    try:
        pred = model.predict(X1)[0]
        proba = None
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(X1)
            if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 2:
                proba = float(p[0,1])
        st.write(f"**Prediction:** {'✅ Survived' if pred==1 else '❌ Did not survive'}")
        if proba is not None:
            st.write(f"**Survival Probability:** {proba*100:.2f}%")
    except Exception as e:
        st.error(f"Error while predicting: {e}")
