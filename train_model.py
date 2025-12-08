import pandas as pd
from xgboost import XGBClassifier
import pickle

print("Loading dataset...")
df = pd.read_csv('data/bnpl_sample_500.csv')

X = df.drop('Default', axis=1)
y = df['Default']

print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
model.fit(X, y)

print("Saving XGBoost model to app/model.pkl...")
with open('app/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
print("You are now ready to deploy to AWS!")
