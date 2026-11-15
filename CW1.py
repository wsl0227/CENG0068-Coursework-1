
from __future__ import annotations
import argparse
from pathlib import Path

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

AMBIENT = ['AT', 'AP', 'AH']  #AT:Ambient Temperature  AP:Ambient Pressure  AH:Ambient Humidity
OPERATING = ['AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'TEY']  #AFDP:Air Filter Difference Pressure  GTEP: Gas Turbine Exhaust Pressure  TIT:Turbine Inlet Temperature TAT:Turbine After Temperature CDP:Compressor Discharge Pressure TEY:Turbine Energy Yield
EMISSIONS = ['CO', 'NOX'] #NOx:	Nitrogen Oxides
COLUMNS = AMBIENT + OPERATING + EMISSIONS
UNIT = {
    'AT':'°C','AP':'mbar','AH':'%','AFDP':'mbar','GTEP':'mbar',
    'TIT':'°C','TAT':'°C','CDP':'mbar','TEY':'MWh',
    'CO':'mg/m³','NOX':'mg/m³'
}
DISPLAY = {'NOX':'NOx'} # "NOX" from original dataset

#--------------------------------------------------------------------#
def display_name(c):
    return f'{DISPLAY.get(c,c)} ({UNIT[c]})'


def load_data():

    folder = Path(__file__).resolve().parent
    files = [folder / f'gt_{year}.csv' for year in range(2011, 2016)]
    df = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True
    )

    return df

    #----------lost data check-----------#
def analyse(df, result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
 
    quality = pd.DataFrame({'missing_n':df.isna().sum(), # is not available
                            'missing_pct':df.isna().mean()*100,
                            'unique_n':df.nunique()})
    quality.to_csv(result_dir/'01_quality.csv')    
    summary = df.describe().T
    summary['IQR'] = summary['75%']-summary['25%']
    summary.to_csv(result_dir/'02_descriptive.csv')

    #--------------IQR--------------#
    flags = []
    for c in COLUMNS:
        q1,q3 = summary.loc[c,['25%','75%']]
        iqr = q3-q1
        low,high = q1-1.5*iqr,q3+1.5*iqr
        flagged = (df[c]<low)|(df[c]>high)
        flags.append({'variable':c,
                      'low_fence':low,
                      'high_fence':high,
                      'outlier_n':int(flagged.sum()),
                      'outlier_pct':100*float(flagged.mean())})
    pd.DataFrame(flags).to_csv(result_dir/'03_IQR_and_flags.csv',index=False)

    #--------------Pearson--------------#
    pearson = df.corr(method='pearson')
    pearson.to_csv(result_dir/'04_Pearson_11.csv')
    pearson.loc[AMBIENT+OPERATING,EMISSIONS].to_csv(result_dir/'05_Pearson_to_Emissions.csv')
    
    #--------------StandardScaler--------------#

    Z = StandardScaler().fit_transform(df)
    Z = pd.DataFrame(Z, columns=df.columns)

    #--------------PCA--------------#
    pca = PCA(n_components=11)
    pca_model = pca.fit(Z)
    df_trans = pd.DataFrame(
        pca_model.transform(Z),
        columns=['PC' + str(i) for i in range(1, 12)],
    )
    individual = pca_model.explained_variance_ratio_ * 100

    variance = pd.DataFrame({
        'PC': ['PC' + str(i) for i in range(1, 12)],
        'individual_pct': individual,
        'cumulative_pct': np.cumsum(individual),
        'eigenvalue': pca_model.explained_variance_
    })
    variance.to_csv(result_dir / '06_PCA_variance.csv')

    eigen = pd.DataFrame(
        pca_model.components_,
        columns=df.columns,
        index=variance['PC']
    )

    eigen.to_csv(result_dir / '07_PCA_eigenvectors.csv')
    score_frame = df_trans[['PC1', 'PC2', 'PC3']]
    return df, summary, pearson, variance, eigen, score_frame


def chart_style():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 9


def save(fig, fig_dir, name):
    fig.savefig(fig_dir / name, dpi=300, bbox_inches='tight')
    plt.close(fig)



    #--------------Fig1 Boxplots--------------#
def plot_boxplots(df, fig_dir):
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))

    for ax, col in zip(axes.flat, COLUMNS):
        ax.boxplot(df[col],flierprops={
        'marker': '.',
        'markersize': 2,
        'markeredgecolor': 'black',
        'alpha': 0.5
    })
        ax.set_title(DISPLAY.get(col, col))
        ax.set_ylabel(UNIT[col])
        ax.set_xticks([])

    axes.flat[-1].axis('off')
    fig.suptitle(f'Figure 1. Boxplots of 11 Variables')
    fig.tight_layout()
    save(fig, fig_dir, 'Figure_1_Boxplots.png')


    #--------------Fig2 Scatter Matrix--------------#
def plot_scatter_matrix(df, fig_dir):
    axes = pd.plotting.scatter_matrix(
        df[COLUMNS],
        figsize=(17, 17),
        diagonal='kde',
        alpha=0.075,
        s=1
    )

    fig = axes[0, 0].figure
    fig.suptitle(f'Figure 2. Scatter Matrix')
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    save(fig, fig_dir, 'Figure_2_Scatter_Matrix.png')


    #--------------Fig3: Pearson Correlation Heatmap--------------#
def plot_pearson(corr, fig_dir, n):
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr,
        cmap='RdBu_r',
        annot=True,
        fmt='.2f',
        vmin=-1,
        vmax=1,
        center=0,
        annot_kws={'size': 8},
        cbar_kws={'label': 'Pearson correlation coefficient (r)'},
        ax=ax
    )

    ax.set_title('Figure 3. Pearson Correlation Matrix')
    fig.tight_layout()

    save(fig, fig_dir, 'Figure_3_Pearson.png')


    #--------------Fig4 PCA Scree Plot--------------#

def plot_scree(variance, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(variance))

    ax.bar(x,variance['individual_pct'],label='Individual Explained Variance')
    ax.plot(x,variance['cumulative_pct'],marker='o',color='orange',label='Cumulative Explained Variance')

    for i, y in enumerate(variance['cumulative_pct']):
        ax.text(x[i],y + 1,f'{y:.1f}%',ha='center',va='bottom',fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(variance['PC'])
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Figure 4. PCA Scree Plot')
    ax.legend()

    fig.tight_layout()
    save(fig, fig_dir, 'Figure_4_PCA_Scree.png')


    #--------------Fig5 PCA Scores and Eigenvectors--------------#

def plot_scores_components(variance, eigen, scores, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(
        scores['PC1'],
        scores['PC2'],
        s=2,
        alpha=0.1
    )

    axes[0].set_xlabel('PC1 Score')
    axes[0].set_ylabel('PC2 Score')
    axes[0].set_title('PC1 vs PC2')

    x = np.arange(11)
    width = 0.25

    axes[1].bar(x - width, eigen.loc['PC1', COLUMNS], width, label='PC1')
    axes[1].bar(        x, eigen.loc['PC2', COLUMNS], width, label='PC2')
    axes[1].bar(x + width, eigen.loc['PC3', COLUMNS], width, label='PC3')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [DISPLAY.get(c, c) for c in COLUMNS],
        rotation=45
    )

    axes[1].set_xlabel('Original Variables')
    axes[1].set_ylabel('Eigenvector Coefficient (unitless)')
    axes[1].set_title('PC1-PC3 Eigenvectors')
    axes[1].legend()

    fig.suptitle('Figure 5. PCA Scores and Eigenvectors')
    fig.tight_layout()
    save(fig, fig_dir, 'Figure_5_PCA_Scores_Eigenvectors.png')




def main():
    root = Path(__file__).resolve().parent

    fig_dir = root / 'figures'
    res_dir = root / 'results'

    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    chart_style()

    df, summary, corr, variance, eigen, scores = analyse(load_data(), res_dir)

    plot_boxplots(df, fig_dir)
    plot_scatter_matrix(df, fig_dir)
    plot_pearson(corr, fig_dir, len(df))
    plot_scree(variance, fig_dir)
    plot_scores_components(variance, eigen, scores, fig_dir)


if __name__ == '__main__':
    main()

