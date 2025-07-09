import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    """Clean text by removing extra whitespace and newlines"""
    if pd.isna(text):
        return ""
    return ' '.join(str(text).split())

def extract_weight_class(weight_str):
    """Extract weight class from weight string"""
    if pd.isna(weight_str) or weight_str == '--':
        return 'Unknown'
    
    weight_str = str(weight_str).lower()
    if '155' in weight_str:
        return 'Lightweight'
    elif '170' in weight_str:
        return 'Welterweight'
    elif '185' in weight_str:
        return 'Middleweight'
    elif '205' in weight_str or '235' in weight_str or '265' in weight_str:
        return 'Heavyweight'
    elif '145' in weight_str:
        return 'Featherweight'
    elif '135' in weight_str:
        return 'Bantamweight'
    elif '125' in weight_str:
        return 'Flyweight'
    else:
        return 'Other'

def calculate_age_at_fight(dob_str, fight_date_str):
    """Calculate fighter's age at the time of the fight"""
    try:
        if pd.isna(dob_str) or pd.isna(fight_date_str) or dob_str == '' or fight_date_str == '':
            return np.nan
        
        dob = pd.to_datetime(dob_str, errors='coerce')
        fight_date = pd.to_datetime(fight_date_str, errors='coerce')
        
        if pd.isna(dob) or pd.isna(fight_date):
            return np.nan
        
        age = (fight_date - dob).days / 365.25
        return age
    except:
        return np.nan

def extract_reach_inches(reach_str):
    """Extract reach in inches from reach string"""
    if pd.isna(reach_str) or reach_str == '--':
        return np.nan
    
    match = re.search(r'(\d+)', str(reach_str))
    if match:
        return int(match.group(1))
    return np.nan

def extract_height_inches(height_str):
    """Extract height in inches from height string"""
    if pd.isna(height_str) or height_str == '--':
        return np.nan
    
    # Handle format like "6' 2""
    match = re.search(r"(\d+)'[ ]?(\d+)?", str(height_str))
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2)) if match.group(2) else 0
        return feet * 12 + inches
    return np.nan

def engineer_features():
    """Engineer features for UFC fight prediction"""
    print("Loading data...")
    
    # Load data
    fighters_df = pd.read_csv('fighters.csv')
    fights_df = pd.read_csv('fights.csv')
    
    print(f"Loaded {len(fighters_df)} fighters and {len(fights_df)} fights")
    
    # Clean the data
    print("Cleaning data...")
    
    # Clean fighters data
    fighters_df['name'] = fighters_df['name'].apply(clean_text)
    fighters_df['height'] = fighters_df['height'].apply(clean_text)
    fighters_df['weight'] = fighters_df['weight'].apply(clean_text)
    fighters_df['reach'] = fighters_df['reach'].apply(clean_text)
    fighters_df['stance'] = fighters_df['stance'].apply(clean_text)
    fighters_df['dob'] = fighters_df['dob'].apply(clean_text)
    
    # Clean fights data
    fights_df['result'] = fights_df['result'].apply(clean_text)
    fights_df['opponent'] = fights_df['opponent'].apply(clean_text)
    fights_df['event'] = fights_df['event'].apply(clean_text)
    fights_df['method'] = fights_df['method'].apply(clean_text)
    fights_df['round'] = fights_df['round'].apply(clean_text)
    fights_df['time'] = fights_df['time'].apply(clean_text)
    
    # Remove fights with missing results
    fights_df = fights_df[fights_df['result'].isin(['win', 'loss', 'draw', 'nc'])]
    
    print(f"After cleaning: {len(fights_df)} fights with valid results")
    
    # Engineer features
    print("Engineering features...")
    
    # 1. Fighter physical features
    fighters_df['weight_class'] = fighters_df['weight'].apply(extract_weight_class)
    fighters_df['reach_inches'] = fighters_df['reach'].apply(extract_reach_inches)
    fighters_df['height_inches'] = fighters_df['height'].apply(extract_height_inches)
    
    # 2. Merge fighter features with fights
    fights_with_features = fights_df.merge(
        fighters_df[['fighter_id', 'weight_class', 'reach_inches', 'height_inches', 'stance', 'dob']], 
        on='fighter_id', 
        how='left'
    )
    
    # 3. Calculate age at fight time
    fights_with_features['age_at_fight'] = fights_with_features.apply(
        lambda row: calculate_age_at_fight(row['dob'], row['date']), axis=1
    )
    
    # 4. Fight-specific features
    fights_with_features['method_category'] = fights_with_features['method'].apply(
        lambda x: 'Submission' if 'sub' in str(x).lower() else 
                 'KO/TKO' if any(term in str(x).lower() for term in ['ko', 'tko', 'knockout']) else
                 'Decision' if 'decision' in str(x).lower() else
                 'Other'
    )
    
    # 5. Calculate fighter statistics (rolling averages)
    print("Calculating fighter statistics...")
    
    # Sort by fighter and date
    fights_with_features['date'] = pd.to_datetime(fights_with_features['date'], errors='coerce')
    fights_with_features = fights_with_features.sort_values(['fighter_id', 'date'])
    
    # Calculate win rate and other stats for each fighter
    fighter_stats = []
    
    for fighter_id in fights_with_features['fighter_id'].unique():
        fighter_fights = fights_with_features[fights_with_features['fighter_id'] == fighter_id].copy()
        
        for idx, fight in fighter_fights.iterrows():
            # Get fights before this one
            previous_fights = fighter_fights[fighter_fights['date'] < fight['date']]
            
            if len(previous_fights) > 0:
                win_rate = (previous_fights['result'] == 'win').mean()
                total_fights = len(previous_fights)
                recent_wins = (previous_fights.tail(3)['result'] == 'win').sum()
                avg_age = previous_fights['age_at_fight'].mean()
            else:
                win_rate = 0.5  # Default for first fight
                total_fights = 0
                recent_wins = 0
                avg_age = fight['age_at_fight']
            
            fighter_stats.append({
                'fight_index': idx,
                'win_rate_before_fight': win_rate,
                'total_fights_before': total_fights,
                'recent_wins': recent_wins,
                'avg_age_before_fight': avg_age
            })
    
    fighter_stats_df = pd.DataFrame(fighter_stats)
    
    # Merge stats back
    final_df = fights_with_features.reset_index().merge(
        fighter_stats_df, left_index=True, right_on='fight_index', how='left'
    )
    
    # 6. Encode categorical variables
    print("Encoding categorical variables...")
    
    le_result = LabelEncoder()
    le_weight_class = LabelEncoder()
    le_stance = LabelEncoder()
    le_method = LabelEncoder()
    
    final_df['result_encoded'] = le_result.fit_transform(final_df['result'])
    final_df['weight_class_encoded'] = le_weight_class.fit_transform(final_df['weight_class'])
    final_df['stance_encoded'] = le_stance.fit_transform(final_df['stance'])
    final_df['method_encoded'] = le_method.fit_transform(final_df['method_category'])
    
    # 7. Select features for modeling
    feature_columns = [
        'age_at_fight', 'reach_inches', 'height_inches', 'weight_class_encoded',
        'stance_encoded', 'win_rate_before_fight', 'total_fights_before',
        'recent_wins', 'avg_age_before_fight'
    ]
    
    # Remove rows with missing values
    final_df_clean = final_df.dropna(subset=feature_columns + ['result_encoded'])
    
    print(f"Final dataset: {len(final_df_clean)} fights with complete features")
    
    return final_df_clean, feature_columns, le_result

def train_model(X, y):
    """Train and evaluate models"""
    print("Training models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # Cross-validation
    rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
    lr_cv_scores = cross_val_score(lr_model, X, y, cv=5)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
    print(f"Random Forest CV Score: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std() * 2:.3f})")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.3f}")
    print(f"Logistic Regression CV Score: {lr_cv_scores.mean():.3f} (+/- {lr_cv_scores.std() * 2:.3f})")
    
    # Feature importance for Random Forest
    feature_names = X.columns
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*50)
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    return rf_model, lr_model, rf_accuracy, lr_accuracy

def main():
    print("UFC Fight Prediction Model")
    print("="*50)
    
    # Engineer features
    final_df, feature_columns, le_result = engineer_features()
    
    # Prepare data for modeling
    X = final_df[feature_columns]
    y = final_df['result_encoded']
    
    print(f"\nFeatures used: {feature_columns}")
    print(f"Target distribution: {final_df['result'].value_counts().to_dict()}")
    
    # Train models
    rf_model, lr_model, rf_acc, lr_acc = train_model(X, y)
    
    # Save the best model
    if rf_acc > lr_acc:
        print(f"\nSaving Random Forest model (accuracy: {rf_acc:.3f})")
        import joblib
        joblib.dump(rf_model, 'ufc_model.pkl')
        joblib.dump(le_result, 'label_encoder.pkl')
        print("Model saved as 'ufc_model.pkl'")
    else:
        print(f"\nSaving Logistic Regression model (accuracy: {lr_acc:.3f})")
        import joblib
        joblib.dump(lr_model, 'ufc_model.pkl')
        joblib.dump(le_result, 'label_encoder.pkl')
        print("Model saved as 'ufc_model.pkl'")

if __name__ == "__main__":
    main() 