import collections
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')
# Drop null rows
loansData.dropna(inplace=True)
# List of column headers
column_names = list(loansData.columns.values)

# Convert data into usable numeric values
# remove '%' from Interest.Rate column
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: x.rstrip('%')).apply(float)
# remove 'months' from Loan.Length column
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: x.rstrip('months')).apply(int)
# replace '-' with ',' in FICO.Range column
loansData['FICO.Score'] = [x.replace('-', ',') for x in loansData['FICO.Range']]
loansData['FICO.Score'] = loansData['FICO.Score'].str.split(',').apply(min).apply(float)
# Save the cleaned data
loansData.to_csv('loansData_clean.csv', header=True, index=False)

plt.figure()
p = loansData['FICO.Score'].hist()
#plt.show()

a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10, 10), diagonal='hist')

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']
# The dependent variable
y = np.matrix(intrate).transpose()

# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1, x2])
# Create the linear model
X = sm.add_constant(x)
model = sm.OLS(y, X)

f = model.fit()
f.summary()
