import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_PATH = Path('processed_data/nonsparsetheta.csv')

def load_data(path: Path) -> pd.DataFrame:
    """Load the processed dataset."""
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

# Binary target mapping
def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary target column."""
    df = df.copy()
    df['y'] = np.where(df['target'] == df['player1_name'], 'player1', 'player2')
    df['maptarget'] = df['y'].map({'player1': 0, 'player2': 1})
    return df

# Feature subset
FEATURE_COLS = [
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

def get_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return encoded feature matrix and target vector."""
    X = df[FEATURE_COLS].copy()
    y = df['maptarget']

    le_surface = LabelEncoder()
    le_level = LabelEncoder()
    X.loc[:, 'surface'] = le_surface.fit_transform(X['surface'])
    X.loc[:, 'tourney_level'] = le_level.fit_transform(X['tourney_level'])
    return X, y


def main() -> None:
    """Run the ML workflow and report accuracy."""
    df = prepare_target(load_data(DATA_PATH))
    X, y = get_features(df)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', LogisticRegression(solver='liblinear')),
    ])

    scoring = ["accuracy", "precision", "recall", "f1"]
    result = cross_validate(
        pipeline,
        X,
        y,
        cv=5,
        scoring=scoring,
        return_train_score=True,
    )
    print("Mean validation accuracy:", result["test_accuracy"].mean())


if __name__ == "__main__":
    main()
