Gas Turbine Emissions – Multivariate Data Analysis

A Python-based exploratory analysis of gas turbine operating conditions and CO/NOₓ emissions, developed for CENG0068 – Fundamentals of Data Science at University College London (UCL).

Dataset

Source: UCI Machine Learning Repository

Dataset: Gas Turbine CO and NOx Emission Data Set

Period: 2011–2015

Observations: 36,733

Variables: 11 numerical parameters covering ambient conditions, turbine operation and pollutant concentrations.

UCI Dataset

Methods

Data preprocessing and outlier identification (IQR)

Boxplots and scatter matrix

Pearson correlation analysis

Data standardisation

Principal Component Analysis (PCA)

Scree plot, PCA scores and eigenvector analysis

Key Findings

Strong correlations exist among turbine operating parameters, particularly pressure and energy yield.

CO concentration is negatively correlated with turbine inlet temperature (r = −0.706).

NOₓ concentration shows its strongest individual correlation with ambient temperature (r = −0.558).

The first three principal components explain 77.35% of the total standardised variance.

Technologies

Python · NumPy · pandas · Matplotlib · Seaborn · scikit-learn
