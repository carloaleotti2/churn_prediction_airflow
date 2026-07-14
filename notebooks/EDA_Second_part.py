import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carica i dati 
df = pd.read_csv('C:/airflow-project/dataset.csv', sep=None, engine='python')
desc = pd.read_csv('C:/airflow-project/data_descriptions.csv', sep=';')
df.head()

# 1. Mappa variabili (nome -> significato)
mapping = dict(zip(desc['Variable'], desc['Significado']))

exclude = [
    'churn', 'Customer_ID',
    'new_cell', 'crclscod', 'asl_flag',
    'prizm_social_one', 'area', 'dualband', 'refurb_new',
    'phones', 'models', 'hnd_webcap', 'truck', 'rv',
    'ownrent', 'dwlltype', 'marital', 'infobase',
    'numbcars', 'HHstatin', 'dwllsize', 'forgntvl', 'ethnic',
    'kid0_2', 'kid3_5', 'kid6_10', 'kid11_15', 'kid16_17',
    'creditcd'
]

num_cols = [c for c in df.columns if c not in exclude]

for col in num_cols:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

missing_pct = df.isna().mean().sort_values(ascending=False)*100

# 3a. Drop colonne con troppi missing (>35%)
cols_to_drop = missing_pct[missing_pct > 35].index.tolist()
print("Drop:", cols_to_drop)
df = df.drop(columns=cols_to_drop)

# 3b. Imputazione categoriche rimanenti con "Missing"
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna('Missing')

# 3c. Imputazione numeriche rimanenti con mediana
num_remaining = df.select_dtypes(include='number').columns
for col in num_remaining:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

from scipy import stats

num_cols = df.select_dtypes(include='number').columns.drop(['churn', 'Customer_ID'], errors='ignore')
cat_cols = df.select_dtypes(include='object').columns

# 1. Correlazione numeriche vs churn
corr_churn = df[num_cols.tolist() + ['churn']].corr()['churn'].sort_values(ascending=False)
print(corr_churn)

# 2. Heatmap correlazioni (top 20 variabili più correlate)
top_corr = corr_churn.abs().sort_values(ascending=False).head(20).index
plt.figure(figsize=(10,8))
sns.heatmap(df[top_corr].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Heatmap correlazioni - top 20 variabili')
plt.tight_layout()
plt.show()

# 3. Boxplot numeriche vs churn (per le top variabili correlate)
for col in top_corr[:10]:
    if col != 'churn':
        plt.figure(figsize=(6,4))
        sns.boxplot(x='churn', y=col, data=df)
        plt.title(f'{col} vs churn')
        plt.show()

# 4. Istogrammi/distribuzioni sovrapposte churn 0 vs 1
for col in top_corr[:10]:
    if col != 'churn':
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x=col, hue='churn', kde=True, stat='density', common_norm=False)
        plt.title(f'Distribuzione {col} per churn')
        plt.show()

# 5. Categoriche vs churn - Chi-square test
chi2_results = {}
for col in cat_cols:
    contingency = pd.crosstab(df[col], df['churn'])
    chi2, p, dof, exp = stats.chi2_contingency(contingency)
    chi2_results[col] = p

chi2_series = pd.Series(chi2_results).sort_values()
print(chi2_series)  # p-value basso = variabile significativa

from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(chi2_series.values, alpha=0.05, method='fdr_bh')

# 6. Barplot categoriche vs churn rate (top variabili significative)
for col in chi2_series.head(10).index:
    plt.figure(figsize=(8,4))
    df.groupby(col)['churn'].mean().sort_values(ascending=False).plot(kind='bar')
    plt.title(f'Churn rate per {col}')
    plt.ylabel('Churn rate')
    plt.show()

# 7. Feature importance rapida con Random Forest (numeriche)
from sklearn.ensemble import RandomForestClassifier
X = df[num_cols].fillna(0)
y = df['churn']
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=num_cols).sort_values(ascending=False)
print(importances.head(20))

plt.figure(figsize=(8,10))
importances.head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Feature importance (Random Forest)')
plt.tight_layout()
plt.show()

from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
perm_importances = pd.Series(perm.importances_mean, index=num_cols).sort_values(ascending=False)