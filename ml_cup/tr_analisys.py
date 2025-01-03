import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento del dataset
data_path = 'dataset/ML-CUP24-TR.csv'
df = pd.read_csv(data_path, header=None, skiprows = 7)

# Aggiunta di nomi alle colonne per chiarezza (prima colonna = indice, le successive = features, le ultime 3 = target)
num_features = 12
num_targets = 3
col_names = ['Index'] + [f'Feature_{i}' for i in range(1, num_features + 1)] + [f'Target_{i}' for i in range(1, num_targets + 1)]
df.columns = col_names

# Rimozione della colonna indice per analisi
data = df.drop(columns=['Index'])

# Analisi descrittiva
print("\nInformazioni generali sul dataset:\n")
print(data.info())

print("\nStatistiche descrittive:\n")
print(data.describe())

# Controllo valori nulli
def check_missing_values(df):
    missing = df.isnull().sum()
    return missing[missing > 0]

print("\nValori mancanti:\n")
print(check_missing_values(data))

# Visualizzazione distribuzioni
plt.figure(figsize=(16, 8))
data.iloc[:, :num_features].hist(bins=30, figsize=(16, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribuzioni delle Features")
plt.show()

# Analisi boxplot per rilevare outlier
plt.figure(figsize=(16, 8))
sns.boxplot(data=data.iloc[:, :num_features], orient="h", palette="coolwarm")
plt.title("Boxplot delle Features per rilevare outlier")
plt.show()

# Calcolo e visualizzazione delle correlazioni
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice di Correlazione")
plt.show()

# Outlier detection basata su IQR
def detect_outliers(df, features):
    outlier_indices = []
    for feature in features:
        Q1 = np.percentile(df[feature], 25)
        Q3 = np.percentile(df[feature], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outliers = df[(df[feature] < Q1 - outlier_step) | (df[feature] > Q3 + outlier_step)].index
        outlier_indices.extend(outliers)
    return list(set(outlier_indices))

outliers = detect_outliers(data, [f'Feature_{i}' for i in range(1, num_features + 1)])
print(f"\nNumero di outlier rilevati: {len(outliers)}")
print(f"Indici degli outlier: {outliers}")

# Visualizzazione distribuzioni dei target
plt.figure(figsize=(16, 8))
data.iloc[:, num_features:].hist(bins=30, figsize=(16, 10), color='orange', edgecolor='black')
plt.suptitle("Distribuzioni dei Target")
plt.show()

# Analisi di skewness e kurtosis per tutte le variabili
from scipy.stats import skew, kurtosis

skewness = data.apply(lambda x: skew(x))
kurt = data.apply(lambda x: kurtosis(x))
print("\nSkewness:\n", skewness)
print("\nKurtosis:\n", kurt)
