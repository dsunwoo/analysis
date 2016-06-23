import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

loansData = pd.read_csv('https://github.com/Thinkful-Ed/'
                        'curric-data-001-data-sets/raw/master/loans/loansData.csv')

# remove rows with null values
loansData.dropna(inplace=True)
# Grab relevant Column data
amt_requested = loansData['Amount.Requested']
amt_funded = loansData['Amount.Funded.By.Investors']
datacols = ['Amount.Requested', 'Amount.Funded.By.Investors']

for col in datacols:
    # Box Plots
    fig = plt.figure()
    loansData.boxplot(column=col)
    fig.suptitle('{}'.format(col), fontsize=20)
    plt.savefig('./graphs/box_{}.png'.format(col))
    # Histograms
    fig = plt.figure()
    loansData.hist(column=col)
    fig.suptitle('{}'.format(col), fontsize=20)
    plt.savefig('./graphs/hist_{}.png'.format(col))
    # Probability Plot
    fig = plt.figure()
    graph = stats.probplot(loansData[col], dist="norm", plot=plt)
    fig.suptitle('{}'.format(col), fontsize=20)
    plt.savefig('./graphs/prob_{}.png'.format(col))
#plt.figure()
#graph = stats.probplot(loansData['Amount.Requested'], dist="norm", plot=plt)
#plt.savefig('loan_prob.png')

