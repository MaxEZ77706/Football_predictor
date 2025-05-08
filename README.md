import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load and prepare data
matches = (
    pd.read_csv('matches.csv', parse_dates=['date'], dayfirst=True)
    .sort_values('date')
    .reset_index(drop=True)
)
# Home/Away flag and target W/D/L
matches['home_flag'] = (matches['venue']=='Home').astype(int)
matches['wdl'] = matches['result'].map({'H':'W','D':'D','A':'L'})

# 2. Compute rolling features (last 5 games) for team and opponent
WINDOW = 5
stats = ['xg','xga','poss','sh','sot','fk','pk','pkatt','dist','attendance']
for s in stats:
    matches[f'{s}_team'] = (
        matches.groupby('team')[s]
               .transform(lambda x: x.shift().rolling(WINDOW, min_periods=1).mean())
    )
    matches[f'{s}_opp'] = (
        matches.groupby('opponent')[s]
               .transform(lambda x: x.shift().rolling(WINDOW, min_periods=1).mean())
    )

# Feature columns
team_cols = [f'{s}_team' for s in stats]
opp_cols  = [f'{s}_opp'  for s in stats]
feature_cols = team_cols + opp_cols + ['home_flag']

# 3. Prepare training data
X = matches[feature_cols].fillna(0)
le = LabelEncoder()
y = le.fit_transform(matches['wdl'])  # 'W','D','L' -> 0,1,2

# 4. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 5. Prepare latest rolling features for prediction
team_latest = (
    matches.groupby('team')[team_cols]
           .last().reset_index()
)
opp_latest = (
    matches.groupby('opponent')[opp_cols]
           .last().reset_index()
)[opp_cols].last().reset_index()


# 6. Predict future matches
upcoming = pd.DataFrame([
    {'team':'Arsenal','opponent':'Bournemouth','home_flag':0},
    {'team':'Chelsea','opponent':'Manchester City','home_flag':1},
])
# Merge features
up = (
    upcoming
    .merge(team_latest, on='team', how='left')
    .merge(opp_latest,  on='opponent', how='left')
)
# Predict W/D/L
X_new = up[feature_cols].fillna(0)
up['pred_WDL'] = le.inverse_transform(model.predict(X_new))
# Determine winner
up['winner'] = up.apply(
    lambda r: r['team'] if r['pred_WDL']=='W' 
              else (r['opponent'] if r['pred_WDL']=='L' else 'Draw'),
    axis=1
)

print(up[['team','opponent','pred_WDL','winner']])
