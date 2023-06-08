import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

train_data = 'data/traindata.csv'
test_data = 'data/testdata.csv'
train_df = pd.read_csv(train_data)

columns = train_df.columns.tolist()
train_df.drop('fnlwgt', axis=1, inplace=True)
mask = train_df.isin(['?'])
rows = mask.any(axis=1)
train_df.drop(train_df[rows].index, inplace=True)
rf_entropy = RandomForestClassifier(criterion='entropy')
rf_gini = RandomForestClassifier(criterion='gini')
rf_entropy.fit(train_data)
rf_gini.fit(train_data)

print(train_df)