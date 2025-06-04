from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional

# Путь к папке с исходными данными
DATA_DIR: Path = Path('data')
PROCESSED_DIR: Path = Path('processed_data')
PROCESSED_DIR.mkdir(exist_ok=True)

# Список всех файлов с матчами ATP
csv_files: list[Path] = sorted(DATA_DIR.glob('atp_matches_*.csv'))

def load_and_concat(files: list[Path]) -> pd.DataFrame:
    """Загружает и объединяет все csv-файлы в один DataFrame."""
    # Все данные будут храниться в этом списке
    dfs: list[pd.DataFrame] = []
    for file in files:
        # Читаем файл
        df: pd.DataFrame = pd.read_csv(file, low_memory=False)
        dfs.append(df)
    # Объединяем все данные
    return pd.concat(dfs, ignore_index=True)

# Загружаем все матчи
all_matches: pd.DataFrame = load_and_concat(csv_files)

# Оставляем только нужные колонки для формирования признаков
base_cols: list[str] = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level', 'tourney_date',
    'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 'winner_rank',
    'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'loser_rank',
]
all_matches = all_matches[base_cols]

# Формируем player1/player2 случайно, чтобы не было утечки
np.random.seed(42)
mask: np.ndarray = np.random.rand(len(all_matches)) > 0.5
player1 = all_matches.loc[mask].assign(
    player1_name=all_matches.loc[mask, 'winner_name'],
    player2_name=all_matches.loc[mask, 'loser_name'],
    player1_ht=all_matches.loc[mask, 'winner_ht'],
    player2_ht=all_matches.loc[mask, 'loser_ht'],
    player1_rank=all_matches.loc[mask, 'winner_rank'],
    player2_rank=all_matches.loc[mask, 'loser_rank'],
    target=all_matches.loc[mask, 'winner_name'],
)
player2 = all_matches.loc[~mask].assign(
    player1_name=all_matches.loc[~mask, 'loser_name'],
    player2_name=all_matches.loc[~mask, 'winner_name'],
    player1_ht=all_matches.loc[~mask, 'loser_ht'],
    player2_ht=all_matches.loc[~mask, 'winner_ht'],
    player1_rank=all_matches.loc[~mask, 'loser_rank'],
    player2_rank=all_matches.loc[~mask, 'winner_rank'],
    target=all_matches.loc[~mask, 'winner_name'],
)

# Объединяем обратно
df = pd.concat([player1, player2], ignore_index=True)

# Признаки h2h, recent_form, theta_form, win_pct — ставим NaN (или можно реализовать позже)
df['player1_h2h'] = np.nan  # TODO: реализовать расчёт h2h
# ... остальные признаки аналогично ...
df['player2_h2h'] = np.nan

df['player_1_recent_form'] = np.nan
# ...
df['player_2_recent_form'] = np.nan

df['player_1_theta_form'] = np.nan
# ...
df['player_2_theta_form'] = np.nan

df['player1_surface_win_pct'] = np.nan
# ...
df['player2_surface_win_pct'] = np.nan

df['player1_level_win_pct'] = np.nan
# ...
df['player2_level_win_pct'] = np.nan

# Оставляем только нужные колонки для ml_workflow.py
final_cols: list[str] = [
    'player1_name', 'player2_name', 'target', 'player1_ht', 'player2_ht',
    'player1_rank', 'player2_rank', 'player1_h2h', 'player2_h2h',
    'surface', 'tourney_level', 'player_1_recent_form', 'player_2_recent_form',
    'player_1_theta_form', 'player_2_theta_form',
    'player1_surface_win_pct', 'player2_surface_win_pct',
    'player1_level_win_pct', 'player2_level_win_pct',
]
df = df[final_cols]

# Сохраняем итоговый датасет
out_path = PROCESSED_DIR / 'nonsparsetheta.csv'
df.to_csv(out_path, index=False)

# Пример Pydantic-модели для одной строки (можно расширить под свои нужды)
class MatchRow(BaseModel):
    # Здесь перечислите нужные поля с типами
    # Например:
    # player1_name: str
    # player2_name: str
    # ...
    pass

# --- Конец скрипта --- 