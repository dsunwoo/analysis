import pandas as pd
from scipy import stats

data = '''Region,Alcohol,Tobacco
North,6.47,4.03
Yorkshire,6.13,3.76
Northeast,6.19,3.77
East Midlands,4.89,3.34
West Midlands,5.63,3.47
East Anglia,4.52,2.92
Southeast,5.89,3.20
Southwest,4.79,2.71
Wales,5.27,3.53
Scotland,6.08,4.51
Northern Ireland,4.02,4.56'''

# Find the end of each line
data = data.splitlines()
# Comma delimited
data = [i.split(',') for i in data]

column_names = data[0]
data_rows = data[1::]
df = pd.DataFrame(data_rows, columns=column_names)

df['Alcohol'] = df['Alcohol'].astype(float)
df['Tobacco'] = df['Tobacco'].astype(float)
# Stats for Alcohol data
al = {}
al['mean'] = df['Alcohol'].mean()
al['median'] = df['Alcohol'].median()
al['mode'] = stats.mode(df['Alcohol'])
al['range'] = max(df['Alcohol']) - min(df['Alcohol'])
al['standard deviation'] = df['Alcohol'].std()
al['variance'] = df['Alcohol'].var()

# Stats for Tobacco data
tb = {}
tb['mean'] = df['Tobacco'].mean()
tb['median'] = df['Tobacco'].median()
tb['mode'] = stats.mode(df['Tobacco'])
tb['range'] = max(df['Tobacco']) - min(df['Tobacco'])
tb['standard deviation'] = df['Tobacco'].std()
tb['variance'] = df['Tobacco'].var()

# Print out results
for k,v in al.items():
    print("The {0} for the Alcohol and Tobacco dataset is: {1} and {2}"
       .format(k, al[k], tb[k]))
