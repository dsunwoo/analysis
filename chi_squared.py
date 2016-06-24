from scipy import stats
import pandas as pd
import collections
import matplotlib.pyplot as plt

# Load the reduced version of the Lending Club Dataset
loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')
# Drop null rows
loansData.dropna(inplace=True)

freq = collections.Counter(loansData['Open.CREDIT.Lines'])

#plt.figure()
#plt.bar(freq.keys(), freq.values(), width=1)
#plt.savefig('./graphs/hist_open_credit_lines')

chi, p = stats.chisquare(list(freq.values()))

print("chi value: {}\np value: {}".format(chi, p))
