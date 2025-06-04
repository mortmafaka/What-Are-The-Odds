import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost
import matplotlib.pyplot as plt
from typing import Any

# Путь к обработанным данным
DATA_PATH: Path = Path('processed_data/nonsparsetheta.csv')

# Загружаем основной датасет и удаляем «Unnamed: 0», если он есть
engin: pd.DataFrame = pd.read_csv(DATA_PATH)
if 'Unnamed: 0' in engin.columns:
    engin = engin.drop('Unnamed: 0', axis=1)

# Преобразуем целевую переменную в бинарный вид и удаляем временный столбец 'y'
engin['y'] = np.where(engin['target'] == engin['player1_name'], 'player1', 'player2')
engin['maptarget'] = engin['y'].map({'player1': 0, 'player2': 1})
engin = engin.drop('y', axis=1)

# Список признаков
feature_cols: list[str] = [
    "player1_ht", "player2_ht", "player1_rank", "player2_rank", "player1_h2h", "player2_h2h",
    "surface", "tourney_level", "player_1_recent_form", "player_2_recent_form",
    "player_1_theta_form", "player_2_theta_form", "player1_surface_win_pct", "player2_surface_win_pct",
    "player1_level_win_pct", "player2_level_win_pct",
]

# Формируем X и y
X: pd.DataFrame = engin[feature_cols].copy()
y: pd.Series = engin['maptarget']

# Кодируем категориальные признаки (surface, tourney_level) отдельно
le_surface = LabelEncoder()
le_level = LabelEncoder()
X.loc[:, 'surface'] = le_surface.fit_transform(X['surface'])
X.loc[:, 'tourney_level'] = le_level.fit_transform(X['tourney_level'])

# Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5323
)

# Pipeline с простым импутером, чтобы не было NaN в признаках, и логистической регрессией
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LogisticRegression(solver='liblinear'))
])

# Кросс-валидация с метриками
scoring = ["accuracy", "precision", "recall", "f1"]
cv_result = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring=scoring,
    return_train_score=True,
)
print("Logistic Regression — средняя валидационная accuracy:", cv_result["test_accuracy"].mean())

# Функция для более детального CV (для RandomForest и XGBoost)
def cv(model: Any, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict[str, Any]:
    results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True
    )
    return {
        "Mean Train Accuracy": results['train_accuracy'].mean() * 100,
        "Mean Train Precision": results['train_precision'].mean(),
        "Mean Train Recall": results['train_recall'].mean(),
        "Mean Train F1": results['train_f1'].mean(),
        "Mean Val Accuracy": results['test_accuracy'].mean() * 100,
        "Mean Val Precision": results['test_precision'].mean(),
        "Mean Val Recall": results['test_recall'].mean(),
        "Mean Val F1": results['test_f1'].mean(),
        "estimators": results['estimator'],
    }

# ===== Random Forest: подбор гиперпараметров через RandomizedSearchCV =====
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)] + [None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [False]

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=10,    # Можно увеличить, если хочется больше проб
    cv=3,
    verbose=2,
    random_state=5323,
    n_jobs=-1
)
rf_random.fit(X_train, y_train)

best_params = rf_random.best_params_
print("Лучшие параметры для RandomForest:", best_params)

# Обучаем случайный лес с подобранными гиперпараметрами
rf_best = RandomForestClassifier(**best_params, n_jobs=-1)
rf_best.fit(X_train, y_train)

# Оцениваем на тестовой выборке
rf_preds = rf_best.predict(X_test)
acc_rf = accuracy_score(y_test, rf_preds)
print("Random Forest accuracy на тесте:", acc_rf)

# ===== Важность признаков из RandomForest =====
plt.figure(figsize=(8, 5))
plt.bar(X_train.columns, rf_best.feature_importances_)
plt.title("Feature Importance (RandomForest)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(Path('processed_data') / 'rf_feature_importance.png')
plt.close()
print("График важности признаков сохранён как 'processed_data/rf_feature_importance.png'.")

# ===== XGBoost =====
xgb = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, xgb_preds)
print("XGBoost accuracy на тесте:", acc_xgb)

# ===== Ансамблирование (majority vote) =====
# Обучаем заново каждый алгоритм на всем X_train
logit = LogisticRegression(solver='liblinear')
rf_ensemble = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    bootstrap=best_params['bootstrap'],
    n_jobs=-1
)
xgb_ensemble = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

logit.fit(X_train, y_train)
rf_ensemble.fit(X_train, y_train)
xgb_ensemble.fit(X_train, y_train)

# Получаем предсказания
logit_pred = logit.predict(X_test)
rf_pred = rf_ensemble.predict(X_test)
xgb_pred = xgb_ensemble.predict(X_test)

# Сборка голосования (majority vote)
stacked_preds = np.vstack([logit_pred, rf_pred, xgb_pred])
ensemble_pred = (stacked_preds.sum(axis=0) >= 2).astype(int)

print("Logistic Regression test accuracy:", accuracy_score(y_test, logit_pred))
print("Random Forest test accuracy:", accuracy_score(y_test, rf_pred))
print("XGBoost test accuracy:", accuracy_score(y_test, xgb_pred))
print("Ensemble (majority vote) test accuracy:", accuracy_score(y_test, ensemble_pred))
