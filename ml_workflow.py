import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost
import matplotlib.pyplot as plt

# Путь к обработанным данным
PROCESSED_DIR: Path = Path('processed_data')

# Загружаем основной датасет
engin: pd.DataFrame = pd.read_csv(PROCESSED_DIR / 'nonsparsetheta.csv')

# Удаляем лишний столбец, если он есть
if 'Unnamed: 0' in engin.columns:
    engin = engin.drop('Unnamed: 0', axis=1)

# Преобразуем целевую переменную в бинарный вид
engin['y'] = np.where(engin['target'] == engin['player1_name'], 'player1', 'player2')
m: dict[str, int] = {"player1": 0, "player2": 1}
engin['maptarget'] = engin['y'].map(m)
engin = engin.drop('y', axis=1)

# Формируем признаки и целевую переменную
feature_cols: list[str] = [
    "player1_ht", "player2_ht", "player1_rank", "player2_rank", "player1_h2h", "player2_h2h",
    "surface", "tourney_level", "player_1_recent_form", "player_2_recent_form",
    "player_1_theta_form", "player_2_theta_form", 'player1_surface_win_pct', 'player2_surface_win_pct',
    'player1_level_win_pct', 'player2_level_win_pct',
]
enginx: pd.DataFrame = engin[feature_cols]
enginy: pd.Series = engin['maptarget']

# Кодируем категориальные признаки
le = LabelEncoder()
enginx['surface'] = le.fit_transform(enginx['surface'])
enginx['tourney_level'] = le.fit_transform(enginx['tourney_level'])

# Делим на train/test
engintrainx, engintestx, engintrainy, engintesty = train_test_split(
    enginx, enginy, test_size=0.25, random_state=5323
)

# Функция для кросс-валидации
from typing import Any

def cv(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 100) -> dict[str, Any]:
    # Считаем метрики по кросс-валидации
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True
    )
    return {
        "Mean Training Accuracy": results['train_accuracy'].mean() * 100,
        "Mean Training Precision": results['train_precision'].mean(),
        "Mean Training Recall": results['train_recall'].mean(),
        "Mean Training F1 Score": results['train_f1'].mean(),
        "Mean Validation Accuracy": results['test_accuracy'].mean() * 100,
        "Mean Validation Precision": results['test_precision'].mean(),
        "Mean Validation Recall": results['test_recall'].mean(),
        "Mean Validation F1 Score": results['test_f1'].mean(),
        "model": results
    }

# Логистическая регрессия
logit = LogisticRegression(solver='liblinear')
logit_result = cv(logit, X=engintrainx, y=engintrainy, cv=5)

# Случайный лес с подбором гиперпараметров
rf = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [False]
randomgrid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=rf, param_distributions=randomgrid, n_iter=2, cv=3, verbose=2, random_state=5323
)
rf_random.fit(engintrainx, engintrainy)

# Обучаем лучший случайный лес
best_params = rf_random.best_params_
rft = RandomForestClassifier(**best_params)
rft.fit(engintrainx, engintrainy)

# Оцениваем точность
preds = rft.predict(engintestx)
acc_rf = accuracy_score(engintesty, preds)

# Важность признаков
plt.bar(engintrainx.columns, rft.feature_importances_)
plt.title("Feature importance per RF model")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / 'rf_feature_importance.png')
plt.close()

# XGBoost
xgb = xgboost.XGBClassifier()
xgb.fit(engintrainx, engintrainy)
xgp = xgb.predict(engintestx)
acc_xgb = accuracy_score(engintesty, xgp)

# Ансамблирование моделей (majority vote)
# Обучаем каждую модель отдельно
logit = LogisticRegression(solver='liblinear')
rf = RandomForestClassifier(
    n_estimators=600,
    min_samples_split=4,
    min_samples_leaf=4,
    max_features='sqrt',
    max_depth=20,
    bootstrap=False
)
xgb = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

logit.fit(engintrainx, engintrainy)
rf.fit(engintrainx, engintrainy)
xgb.fit(engintrainx, engintrainy)

# Предсказания каждой модели
logit_pred = logit.predict(engintestx)
rf_pred = rf.predict(engintestx)
xgb_pred = xgb.predict(engintestx)

# Ансамбль: голосование большинства
pred_stack = np.vstack([logit_pred, rf_pred, xgb_pred])
ensemble_pred = (pred_stack.sum(axis=0) >= 2).astype(int)

# Выводим точности
print('Logistic Regression accuracy:', accuracy_score(engintesty, logit_pred))
print('Random Forest accuracy:', accuracy_score(engintesty, rf_pred))
print('XGBoost accuracy:', accuracy_score(engintesty, xgb_pred))
print('Ensemble accuracy:', accuracy_score(engintesty, ensemble_pred))

# --- Конец скрипта --- 