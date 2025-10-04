
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parents[0]))
import config
import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import itertools
import math


df = DataLoader.LoadKNN()
print(df.columns)


def prepare(df, cols =[], normalize = False):
    df = df.drop(columns=["name", "session","color"], errors="ignore")  
    if len(cols) > 0:
        df = df[cols]
    if normalize:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        num_cols = df.select_dtypes(include=[np.number]).columns
        df["Summe"] = df[num_cols].sum(axis=1)
        for col in num_cols:
            df[col] = df[col] / df["Summe"] 
    df.drop(columns=["Summe"], errors="ignore", inplace=True)
    return df

def prep_norm(df):
    df = df.drop(columns=["name", "session", "color", "tier1", "tier2", "tier3", "building"], errors="ignore")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include=[np.number]).columns
    df["Summe"] = df[num_cols].sum(axis=1)
    for col in num_cols:
        df[col + "_prop"] = df[col] / df["Summe"]

    feature_cols_prop = [col + "_prop" for col in num_cols]
    X_prop = df[feature_cols_prop].copy()
    return X_prop 
    
    
        
data = prep_norm(df)

def cluster_kmeans(K,cols=['air', 'land', 'naval']):
    data = prepare(df,cols = cols, normalize = True)
    columns = data.columns.tolist()
    scaler = StandardScaler()

    kmeans = KMeans(n_clusters=K, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    centers_df = pd.DataFrame(centers, columns=columns)
    centers_df.index.name = 'Cluster'
    print(f"centers_df:\n{centers_df}")
    return data, centers_df

def standard_plot(K=3,colss=['air', 'land', 'naval']):
    data, centers_df = cluster_kmeans(K=K, cols=colss)
    columns = data.columns.tolist()
    print("=============== PLOT ===============")


    combinations = list(itertools.combinations(columns, 2))
    combinations = [c for c in combinations if "cluster" not in c[0].lower() and "cluster" not in c[1].lower()]
    num_coms = len(combinations)
    print(f"Kombinationen: {combinations}")


    num_coms = len(combinations)
    cols = 3
    rows = math.ceil(num_coms / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, num_coms*2)) # Erstellt 1 Zeile, 3 Spalten
    for i, (feat_x, feat_y) in enumerate(combinations):
        if cols <= 3:
            ax = axes[i % cols]
        else:
            ax = axes[i // cols, i % cols]
        # Scatter-Plot der Datenpunkte
        ax.scatter(data[feat_x], data[feat_y],
                    c=data['Cluster'],
                    cmap='viridis',
                    s=50,
                    alpha=0.8,
                    edgecolor='k')

        # Plotten der Cluster-Zentren
        ax.scatter(centers_df[feat_x], centers_df[feat_y],
                    marker='X',
                    s=100,
                    color='red',
                    label='Zentren',
                    edgecolor='black')

        ax.set_xlabel(feat_x)
        ax.set_ylabel(feat_y)
        ax.set_title(f'{feat_x} vs. {feat_y}')
        
        # Zeigt die Legende nur im ersten Plot an
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close()



def prev_plot():
    ...



standard_plot(K=3, colss = ['air', 'land', 'naval'])
#'air', 'land', 'naval', 'building', 'tier1', 'tier2', 'tier3', 'color','name', 'session'



