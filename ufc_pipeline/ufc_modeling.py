import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
import re
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import shap

# --- Helper functions to parse height, weight, reach ---
def parse_height(height_str):
    if not isinstance(height_str, str):
        return None
    match = re.match(r"^(\d+)' (\d+)\"", height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return feet * 12 + inches
    return None

def parse_weight(weight_str):
    if pd.isna(weight_str):
        return None
    match = re.match(r"(\d+)", str(weight_str))
    if match:
        return float(match.group(1))
    try:
        return float(weight_str)
    except:
        return None

def parse_reach(reach_str):
    if pd.isna(reach_str):
        return None
    match = re.match(r"(\d+)", str(reach_str))
    if match:
        return float(match.group(1))
    try:
        return float(reach_str)
    except:
        return None

def compute_last_n_win_rate(df, fighter_col, date_col, result_col, n=10):
    df = df.sort_values([fighter_col, date_col])
    last_n_win_rate = []
    for fighter_id, group in df.groupby(fighter_col):
        results = group[result_col].tolist()
        rates = []
        for i in range(len(results)):
            if i < n:
                rates.append(np.nan)  # Not enough history
            else:
                rates.append(np.mean(results[i-n:i]))
        last_n_win_rate.extend(rates)
    return last_n_win_rate

# --- Elo rating computation ---
def compute_elo_ratings(df, k=32, base=1500):
    """
    Compute Elo ratings for all fighters, updating after each fight in chronological order.
    Returns a DataFrame with pre-fight Elo for fighter and opponent, and Elo diff.
    """
    # Sort by date (temporal order)
    df = df.sort_values('date').reset_index(drop=True)
    # Map fighter_id to Elo
    elo_dict = {}
    pre_elo_f, pre_elo_o = [], []
    for idx, row in df.iterrows():
        f_id = row['fighter_id']
        o_id = row['opponent_id']
        # Use base Elo if new
        f_elo = elo_dict.get(f_id, base)
        o_elo = elo_dict.get(o_id, base)
        pre_elo_f.append(f_elo)
        pre_elo_o.append(o_elo)
        # Only update if both IDs present
        if pd.notnull(f_id) and pd.notnull(o_id):
            # 1 if fighter wins, 0 if loses
            result = row['label']
            # Expected scores
            ef = 1 / (1 + 10 ** ((o_elo - f_elo) / 400))
            eo = 1 / (1 + 10 ** ((f_elo - o_elo) / 400))
            # Update Elo
            elo_dict[f_id] = f_elo + k * (result - ef)
            elo_dict[o_id] = o_elo + k * ((1 - result) - eo)
    df['f_elo'] = pre_elo_f
    df['o_elo'] = pre_elo_o
    df['elo_diff'] = df['f_elo'] - df['o_elo']
    return df

print("🥊 UFC Fight Prediction Modeling")
print("="*50)

# Load data
fighters = pd.read_csv('fighters.csv')
fights = pd.read_csv('fights.csv')

# --- After loading fights.csv ---

# Print missing rates for key features in fighters.csv

# --- Print columns and first 10 rows for diagnosis ---

# --- 1. Deduplicate: one row per fight per fighter ---
# Drop duplicates (in case of rematches on same event/date/fighter_id/opponent)
before_dedup = len(fights)
fights = fights.drop_duplicates(subset=["event", "date", "fighter_id", "opponent_id"])

# --- TEMP FIX: Extract last two words from 'opponent' as cleaned name ---
def extract_last_two_names(s):
    if not isinstance(s, str):
        return s
    parts = s.split()
    return ' '.join(parts[-2:]) if len(parts) >= 2 else s
fights['opponent_clean'] = fights['opponent'].apply(extract_last_two_names)

# --- After deduplication/merge (immediately after deduplication logic, before merging features) ---

# Prepare fighter features
df_fighters = fighters.rename(columns={'fighter_id': 'id'})
df_fighters['id'] = df_fighters['id'].astype(str)
df_fighters = df_fighters.set_index('id')

# --- Parse height, weight, reach for numeric modeling ---
for prefix in ['','o_','f_']:
    for col, func in zip(['height','weight','reach'], [parse_height, parse_weight, parse_reach]):
        colname = f'{prefix}{col}' if prefix else col
        if colname in df_fighters.columns:
            df_fighters[colname] = df_fighters[colname].apply(func)

# --- After merging fighter/opponent features (immediately after both joins) ---

# --- Ensure IDs are strings before merging ---
fights['fighter_id'] = fights['fighter_id'].astype(str)
fights['opponent_id'] = fights['opponent_id'].astype(str)
df_fighters.index = df_fighters.index.astype(str)

# Merge fighter features (prefix 'f_')
fights = fights.join(df_fighters.add_prefix('f_'), on='fighter_id')

# Merge opponent features (prefix 'o_')
fights = fights.join(df_fighters.add_prefix('o_'), on='opponent_id')

# --- Explicitly cast numeric columns to float ---
for col in ['f_height', 'f_weight', 'f_reach', 'o_height', 'o_weight', 'o_reach']:
    if col in fights.columns:
        fights[col] = pd.to_numeric(fights[col], errors='coerce')

# --- Fallback: If opponent features are still missing, merge by cleaned name ---
missing = fights[['o_name', 'o_height', 'o_weight']].isna().mean()
if missing.max() > 0.9:
    fights = fights.drop(['o_name', 'o_height', 'o_weight', 'o_reach', 'o_stance', 'o_dob'], axis=1, errors='ignore')
    fights = fights.merge(fighters.add_prefix('o_'), left_on='opponent_clean', right_on='o_name', how='left')
    # Parse merged opponent columns to numeric
    for col, func in zip(['o_height', 'o_weight', 'o_reach'], [parse_height, parse_weight, parse_reach]):
        if col in fights.columns:
            fights[col] = fights[col].apply(func)

# After merge by cleaned name

# --- Compute win rate for each fighter up to each fight ---
def compute_win_rate(df, fighter_col, date_col, result_col):
    df = df.sort_values([fighter_col, date_col])
    win_rates = []
    for fighter_id, group in df.groupby(fighter_col):
        wins = 0
        total = 0
        rates = []
        for idx, row in group.iterrows():
            if total == 0:
                rates.append(0.5)  # Default for first fight
            else:
                rates.append(wins / total)
            if row[result_col] == 1:
                wins += 1
            total += 1
        win_rates.extend(rates)
    return win_rates

# Compute win rate for 'fighter' and 'opponent' perspectives
fights['f_win_rate'] = compute_win_rate(fights, 'fighter_id', 'date', 'result')
fights['o_win_rate'] = compute_win_rate(fights, 'opponent_id', 'date', 'result')

# --- Diagnostics before core-drop ---

# --- Run missingness report ---

# ---------------------------------------------------------------
# 5.  LABEL MAPPING  (Win=1  •  Loss=0  •  drop Draw / NC)
valid_results = {"win": 1, "loss": 0, "Win": 1, "Loss": 0}
before_lbl = len(fights)

fights = (fights
          .assign(label = fights["result"].map(valid_results))
          .dropna(subset=["label"])
          .astype({"label": "int8"}))          # label now 0/1

# --- Ensure 'date' is datetime before feature engineering ---
fights['date'] = pd.to_datetime(fights['date'], errors='coerce')
# --- Add fight recency (days since last fight) ---
fights = fights.sort_values(["date"])
fights["f_days_since"] = (
    fights.groupby("fighter_id")['date'].diff().dt.days.clip(lower=0)
)
fights["o_days_since"] = (
    fights.groupby("opponent_id")['date'].diff().dt.days.clip(lower=0)
)
fights["f_days_since"].fillna(365, inplace=True)
fights["o_days_since"].fillna(365, inplace=True)

# --- Add 5-fight win streak indicator ---
fights["f_win_streak5"] = (
    fights.groupby("fighter_id")["label"]
    .rolling(5, closed="left").sum().reset_index(level=0, drop=True).fillna(0)
)
fights["o_win_streak5"] = (
    fights.groupby("opponent_id")["label"]
    .rolling(5, closed="left").sum().reset_index(level=0, drop=True).fillna(0)
)

# --- Add is_title_fight flag ---
fights['is_title_fight'] = fights['event'].str.contains('Title', case=False, na=False).astype(int)

# --- Compute finish, KO, SUB rates, and total fights for both fighter and opponent ---
def compute_stats(df, fighter_col, date_col, result_col, method_col):
    df = df.sort_values([fighter_col, date_col])
    finish_rate = []
    ko_rate = []
    sub_rate = []
    total_fights = []
    for fighter_id, group in df.groupby(fighter_col):
        finishes = 0
        kos = 0
        subs = 0
        total = 0
        finish_rates = []
        ko_rates = []
        sub_rates = []
        totals = []
        for idx, row in group.iterrows():
            if total == 0:
                finish_rates.append(0.5)
                ko_rates.append(0.25)
                sub_rates.append(0.25)
                totals.append(0)
            else:
                finish_rates.append(finishes / total)
                ko_rates.append(kos / total)
                sub_rates.append(subs / total)
                totals.append(total)
            if row[result_col] == 1:
                if isinstance(row[method_col], str):
                    m = row[method_col].lower()
                    if any(x in m for x in ['ko', 'tko', 'knockout']):
                        kos += 1
                        finishes += 1
                    elif 'sub' in m:
                        subs += 1
                        finishes += 1
                    elif 'decision' not in m:
                        finishes += 1
            total += 1
        finish_rate.extend(finish_rates)
        ko_rate.extend(ko_rates)
        sub_rate.extend(sub_rates)
        total_fights.extend(totals)
    return finish_rate, ko_rate, sub_rate, total_fights

fights['f_finish_rate'], fights['f_ko_rate'], fights['f_sub_rate'], fights['f_total_fights'] = compute_stats(fights, 'fighter_id', 'date', 'label', 'method')
fights['o_finish_rate'], fights['o_ko_rate'], fights['o_sub_rate'], fights['o_total_fights'] = compute_stats(fights, 'opponent_id', 'date', 'label', 'method')

# --- Print missingness for new features ---

# --- Compute Elo ratings for the entire dataset ---
fights_with_elo = compute_elo_ratings(fights)

# --- Add Elo features to the feature set for modeling ---
feature_cols = [
    # core features
    'f_height', 'o_height', 'f_weight', 'o_weight',
    'f_win_rate', 'o_win_rate',
    'f_finish_rate', 'o_finish_rate',
    'f_ko_rate', 'o_ko_rate',
    'f_sub_rate', 'o_sub_rate',
    'f_total_fights', 'o_total_fights',
    'height_diff', 'weight_diff', 'win_rate_diff', 'finish_rate_diff',
    'ko_rate_diff', 'sub_rate_diff', 'total_fights_diff',
    'days_since_last_fight', 'is_title_fight', 'five_fight_win_streak',
    # Elo features
    'elo_diff', 'f_elo', 'o_elo',
]

# --- Feature engineering: ensure all features are present before splitting ---
fights_with_elo['height_diff'] = fights_with_elo['f_height'] - fights_with_elo['o_height']
fights_with_elo['weight_diff'] = fights_with_elo['f_weight'] - fights_with_elo['o_weight']
fights_with_elo['win_rate_diff'] = fights_with_elo['f_win_rate'] - fights_with_elo['o_win_rate']
fights_with_elo['finish_rate_diff'] = fights_with_elo['f_finish_rate'] - fights_with_elo['o_finish_rate']
fights_with_elo['ko_rate_diff'] = fights_with_elo['f_ko_rate'] - fights_with_elo['o_ko_rate']
fights_with_elo['sub_rate_diff'] = fights_with_elo['f_sub_rate'] - fights_with_elo['o_sub_rate']
fights_with_elo['total_fights_diff'] = fights_with_elo['f_total_fights'] - fights_with_elo['o_total_fights']
# Elo features already added by compute_elo_ratings

# Map recency and streak features to expected column names for modeling
fights_with_elo['days_since_last_fight'] = fights_with_elo['f_days_since']
fights_with_elo['five_fight_win_streak'] = fights_with_elo['f_win_streak5']

# Add last-10 win rate features
fights_with_elo['f_last10_win_rate'] = compute_last_n_win_rate(fights_with_elo, 'fighter_id', 'date', 'label', n=10)
fights_with_elo['o_last10_win_rate'] = compute_last_n_win_rate(fights_with_elo, 'opponent_id', 'date', 'label', n=10)
fights_with_elo['last10_win_rate_diff'] = fights_with_elo['f_last10_win_rate'] - fights_with_elo['o_last10_win_rate']

# ---------------------------------------------------------------
# 6.  TEMPORAL TRAIN / TEST SPLIT  (adjust cutoff as needed)
cutoff = pd.Timestamp("2022-01-01")
train = fights_with_elo[fights_with_elo["date"] <  cutoff]
test  = fights_with_elo[fights_with_elo["date"] >= cutoff]

# 1. drop rows lacking any of the core four
before = len(fights_with_elo)
fights_with_elo  = fights_with_elo.dropna(subset=feature_cols)

# 2. impute every remaining numeric column (reach, age, etc.)
from sklearn.impute import SimpleImputer
num_cols = fights_with_elo.select_dtypes(include="number").columns
imputer  = SimpleImputer(strategy="median")
fights_with_elo[num_cols] = imputer.fit_transform(fights_with_elo[num_cols])

# --- Add new features to feature_cols ---
# feature_cols = [
#     'f_height', 'f_weight', 'f_reach', 'f_stance', 'f_age',
#     'o_height', 'o_weight', 'o_reach', 'o_stance', 'o_age',
#     'f_win_rate', 'o_win_rate', 'win_rate_diff',
#     'f_last10_win_rate', 'o_last10_win_rate', 'last10_win_rate_diff',
# ]
# feature_cols += ["f_days_since", "o_days_since", "f_win_streak5", "o_win_streak5"]
# feature_cols += [
#     'is_title_fight',
#     'f_finish_rate', 'f_ko_rate', 'f_sub_rate', 'f_total_fights',
#     'o_finish_rate', 'o_ko_rate', 'o_sub_rate', 'o_total_fights'
# ]
# feature_cols += ["f_last10_win_rate", "o_last10_win_rate", "last10_win_rate_diff"]

# Drop rows with missing features
train = train.dropna(subset=feature_cols)
test = test.dropna(subset=feature_cols)

# Encode categorical features (stance)
for col in ['f_stance', 'o_stance']:
    train[col] = train[col].astype('category').cat.codes
    test[col] = test[col].astype('category').cat.codes

# Convert dates of birth to age at fight time
def compute_age(dob, fight_date):
    try:
        dob = pd.to_datetime(dob, errors='coerce')
        return (fight_date - dob).days / 365.25
    except:
        return None
fights_with_elo['f_age'] = fights_with_elo.apply(lambda row: compute_age(row['f_dob'], row['date']), axis=1)
fights_with_elo['o_age'] = fights_with_elo.apply(lambda row: compute_age(row['o_dob'], row['date']), axis=1)

# Drop rows with missing features again
train = train.dropna(subset=feature_cols)
test = test.dropna(subset=feature_cols)

# If test set is empty, suggest a new cutoff date
if len(test) == 0:
    print('Test set is empty. Try using an earlier cutoff date, e.g., 2020-01-01.')

# --- Build feature matrix and target variable correctly ---
X_train = train[feature_cols]
y_train = train['label']
X_test = test[feature_cols]
y_test = test['label']

# --- Before model training, print class balance in y_train and y_test ---

# --- Hyperparameter tuning for XGBoost ---
param_dist = {
    'n_estimators': [200, 400, 600],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}
xgb_base = XGBClassifier(eval_metric='auc', use_label_encoder=False)
search = RandomizedSearchCV(
    xgb_base, param_distributions=param_dist, n_iter=12, scoring='roc_auc',
    cv=3, verbose=2, random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)

# --- Evaluate best model on test set ---
xgb_best = search.best_estimator_
y_pred = xgb_best.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)

# --- Probability calibration (sigmoid) ---
calib = CalibratedClassifierCV(search.best_estimator_, method='sigmoid', cv='prefit')
calib.fit(X_train, y_train)
y_pred_calib = calib.predict_proba(X_test)[:, 1]
calib_auc = roc_auc_score(y_test, y_pred_calib)
calib_brier = brier_score_loss(y_test, y_pred_calib)

print("\n🎉 Modeling complete! Ready for predictions.")

# --- Drop rows with missing features (strict: all features) ---
train_strict = train.dropna(subset=feature_cols)
test_strict = test.dropna(subset=feature_cols)

# --- Relaxed: Only require height and weight, impute others ---
relaxed_cols = ['f_height', 'f_weight', 'o_height', 'o_weight']
train_relaxed = train.dropna(subset=relaxed_cols)
test_relaxed = test.dropna(subset=relaxed_cols)

# Impute missing values for other features
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
for df in [train_relaxed, test_relaxed]:
    for col in feature_cols:
        if col not in relaxed_cols and col in df.columns:
            df[col] = imputer.fit_transform(df[[col]])

# --- If test set is empty, try random split ---
if len(test_strict) == 0:
    print('Test set is empty after strict dropna. Trying random split...')
    from sklearn.model_selection import train_test_split
    all_data = fights.dropna(subset=feature_cols)
    train_rand, test_rand = train_test_split(all_data, test_size=0.2, random_state=42)
    print(f"Rows in train (random split): {len(train_rand)}")
    print(f"Rows in test (random split): {len(test_rand)}")
    print('Value counts for result in train (random split):')
    print(train_rand['result'].value_counts())
    print('Value counts for result in test (random split):')
    print(test_rand['result'].value_counts())

# --- Print value counts for all splits ---
print('Value counts for result in train (strict):')
print(train_strict['result'].value_counts())
print('Value counts for result in test (strict):')
print(test_strict['result'].value_counts())
print('Value counts for result in train (relaxed):')
print(train_relaxed['result'].value_counts())
print('Value counts for result in test (relaxed):')
print(test_relaxed['result'].value_counts())

print(f"Final number of rows for modeling: {len(train_strict) + len(test_strict)} (strict), {len(train_relaxed) + len(test_relaxed)} (relaxed)")
print(f"Unique fights (strict): {train_strict[['event','date','fighter_id','opponent_id']].drop_duplicates().shape[0] + test_strict[['event','date','fighter_id','opponent_id']].drop_duplicates().shape[0]}")
print(f"Unique fighters (strict): {len(set(train_strict['fighter_id']).union(set(test_strict['fighter_id'])))}")
print(f"Unique fights (relaxed): {train_relaxed[['event','date','fighter_id','opponent_id']].drop_duplicates().shape[0] + test_relaxed[['event','date','fighter_id','opponent_id']].drop_duplicates().shape[0]}")
print(f"Unique fighters (relaxed): {len(set(train_relaxed['fighter_id']).union(set(test_relaxed['fighter_id'])))}") 

# --- SHAP analysis for XGBoost ---
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_test)

# Summary bar plot
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig("shap_summary_bar.png")
plt.close()

# Beeswarm plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png")
plt.close()

print("SHAP summary plots saved as 'shap_summary_bar.png' and 'shap_summary_beeswarm.png'") 

import pandas as pd
pd.set_option('display.max_rows', None)
shap_df = pd.DataFrame({
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}, index=X_test.columns).sort_values('mean_abs_shap', ascending=False)
print("\nFull SHAP feature importance table:")
print(shap_df)
shap_df.to_csv('shap_feature_importance.csv')
print("SHAP feature importance table saved as 'shap_feature_importance.csv'") 
win_features = [
    'f_win_rate', 'o_win_rate', 'win_rate_diff',
    'f_last10_win_rate', 'o_last10_win_rate', 'last10_win_rate_diff'
]
print("\nSHAP values for win-related features:")
print(shap_df.loc[shap_df.index.intersection(win_features)]) 

from sklearn.metrics import accuracy_score
print(f"Test AUC: {auc:.3f}")
print(f"Test Accuracy: {accuracy_score(y_test, xgb_best.predict(X_test)):.3f}") 