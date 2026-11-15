<div align="center">

Gas Turbine Emissions Analysis

Multivariate Statistical Analysis of Operating Conditions and CO/NOₓ Emissions

CENG0068 · Fundamentals of Data Science · UCL





Exploring relationships between turbine operating conditions and pollutant emissions through statistical analysis and dimensionality reduction.

</div>

Overview

This project investigates the relationships between ambient conditions, gas turbine operating parameters, and CO/NOₓ emissions using exploratory multivariate statistical methods.

The analysis focuses on variable correlations, potential outliers, information redundancy, and principal component analysis (PCA).

Dataset

Property

Description

Source

UCI Machine Learning Repository

Dataset

Gas Turbine CO and NOx Emission Data Set

Period

2011–2015

Observations

36,733

Variables

11 numerical parameters

Categories

Ambient, Operating, Emissions

Dataset DOI: https://doi.org/10.24432/C5WC95

Analysis Workflow

Data Collection → Preprocessing → Visualisation → Correlation Analysis → PCA → Interpretation

The project includes:

Boxplots and IQR-based outlier identification

Scatter matrix for pairwise relationships

Pearson correlation heatmap

Standardisation and principal component analysis

PCA scree plot, scores and eigenvector interpretation

Key Findings

Analysis

Result

CDP vs TEY

r = 0.989

TIT vs CO

r = −0.706

AT vs NOₓ

r = −0.558

PCA – First 2 Components

67.90% explained variance

PCA – First 3 Components

77.35% explained variance

The results reveal strong correlations among turbine operating parameters and distinct statistical associations for CO and NOₓ emissions.

Visualisations

The analysis produces five figures:

Figure

Description

Figure 1

Boxplots of 11 Variables

Figure 2

Scatter Matrix

Figure 3

Pearson Correlation Matrix

Figure 4

PCA Scree Plot

Figure 5

PCA Scores and Eigenvectors

Tools and Libraries

Python · NumPy · pandas · Matplotlib · Seaborn · scikit-learn

Reference

UCI Machine Learning Repository. Gas Turbine CO and NOx Emission Data Set (2019).

DOI: https://doi.org/10.24432/C5WC95

<div align="center">

University College London · Fundamentals of Data Science

</div>
