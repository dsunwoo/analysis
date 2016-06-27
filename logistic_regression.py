import pandas as pd
import statsmodels.api as sm
import math


def logistic_function(fscore, loan):
    pvalue = 1/(1 + math.exp(coeff[2] + (coeff[1] * fscore) - (coeff[0] * loan)))
    print("Loan amount: {0} \nFICO score: {1} \nProb. score: {2}".format(loan, fscore, pvalue))
    pred(pvalue)
    return pvalue

def pred(pvalue):
    if pvalue >= 0.7:
        print("Interest Rate: <=12%")
    else:
        print("Interest Rate: >12%")

df = pd.read_csv('loansData_clean.csv')
df['IR_TF'] = df['Interest.Rate'].map(lambda x: True if x >= 12 else False)
df['Intercept'] = df['Interest.Rate'].map(lambda x: 1).apply(float)
# List of column headers
ind_vars = ['Amount.Requested', 'FICO.Score', 'Intercept']

logit = sm.Logit(df['IR_TF'], df[ind_vars])
result = logit.fit()
coeff = result.params

p = logistic_function(720, 10000)
