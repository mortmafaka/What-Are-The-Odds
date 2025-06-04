import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load processed dataset
DF_PATH = 'processed_data/nonsparsetheta.csv'

# Read data and drop index column if present
engin = pd.read_csv(DF_PATH)
if 'Unnamed: 0' in engin.columns:
    engin = engin.drop('Unnamed: 0', axis=1)

# Binary target mapping
engin['y'] = np.where(engin['target'] == engin['player1_name'], 'player1', 'player2')
engin['maptarget'] = engin['y'].map({'player1': 0, 'player2': 1})

# Feature subset
feature_cols = [
    'player1_ht',
    'player2_ht',
    'player1_rank',
    'player2_rank',
    'player1_h2h',
    'player2_h2h',
    'surface',
    'tourney_level',
    'player_1_recent_form',
    'player_2_recent_form',
    'player_1_theta_form',
    'player_2_theta_form',
    'player1_surface_win_pct',
    'player2_surface_win_pct',
    'player1_level_win_pct',
    'player2_level_win_pct'
]
# Use .copy() to avoid pandas SettingWithCopyWarning
X = engin[feature_cols].copy()
Y = engin['maptarget']

# Encode categorical columns using separate encoders
le_surface = LabelEncoder()
le_level = LabelEncoder()
X.loc[:, 'surface'] = le_surface.fit_transform(X['surface'])
X.loc[:, 'tourney_level'] = le_level.fit_transform(X['tourney_level'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=5323)

# Pipeline with imputation to handle any missing values
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LogisticRegression(solver='liblinear'))
])

scoring = ["accuracy", "precision", "recall", "f1"]
result = cross_validate(pipeline, X_train, y_train, cv=5, scoring=scoring, return_train_score=True)
print("Mean validation accuracy:", result["test_accuracy"].mean())
