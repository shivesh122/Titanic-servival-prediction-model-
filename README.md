# Titanic Survival Prediction (Streamlit)

Fully deployable Streamlit app with:
- Batch CSV prediction
- Single passenger prediction
- Model saved and loaded dynamically
- Ready for GitHub → Vercel deployment (or Streamlit Community Cloud)

## Project structure

```
titanic_app/
├── models/
│   └── titanic_best_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Vercel (quick notes)

1. Push this folder to a public GitHub repo.
2. On Vercel: “Import Git Repository” → select your repo.
3. Framework Preset: **Other**.
4. Build Command: `streamlit run app.py`
5. Output Directory: leave blank.
6. Deploy.

> Tip: Some users prefer Streamlit Community Cloud for zero-config Streamlit hosting.

## CSV template

Download a CSV with the typical Titanic columns from inside the app (Batch section). Make sure your CSV has headers similar to Kaggle Titanic:  
`PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked`

## Model

For a frictionless start, this repo ships with a lightweight heuristic model implementing `predict` and `predict_proba`. You can replace it with your trained model (pickle at `models/titanic_best_model.pkl`). It should accept a pandas DataFrame containing the engineered feature columns:

`['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','IsAlone','Deck','FarePerPerson']`

