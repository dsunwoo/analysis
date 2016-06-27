import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('loansData_clean.csv')
# Add annual income column based on Monthly.Income
df['Annual_Income'] = df['Monthly.Income'].map(lambda x: x*12)
df['Intercept'] = float(1.0)
# Rename columns to make smf function work
df['Interest_Rate'] = df['Interest.Rate']
df['Home_Ownership'] = df['Home.Ownership']
# column_names = list(df.columns.values)

# Interest.Rate ~ Annual.Income
est1 = smf.ols(formula='Interest_Rate ~ Annual_Income', data=df).fit()
print("\nInterest_Rate ~ Annual_Income Model\n{}".format(est1.summary()))

# Add Home.Ownership to the model
est2 = smf.ols(formula='Interest_Rate ~ Annual_Income + Home_Ownership', data=df).fit()
print("\nInterest_Rate ~ Annual_Income + Home_Ownership Model\n{}".format(est2.summary()))

# Model the interaction of Annual_Income and Home_Ownership
est3 = smf.ols(formula='Interest_Rate ~ Annual_Income * Home_Ownership', data=df).fit()
print("\nInterest_Rate ~ Annual_Income * Home_Ownership Model\n{}".format(est3.summary()))
