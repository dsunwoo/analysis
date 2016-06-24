import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('loansData_clean.csv')
df['IR_TF'] = df['Interest.Rate'].map(lambda x: 1 if x <= 12 else 0)
df['Intercept'] = df['Interest.Rate'].map(lambda x: 1)
# List of column headers
ind_vars = list(df.columns.values)

logit = sm.Logit(df['IR_TF'], df[ind_vars])
result = logit.fit()
