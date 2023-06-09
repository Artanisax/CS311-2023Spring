import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier

# read files
data_file = 'data/traindata.csv'
lable_file = 'data/trainlabel.txt'
judge_file = 'data/testdata.csv'
data_df = pd.read_csv(data_file)
judge_df = pd.read_csv(judge_file)
lable = []
with open(lable_file) as f:
    for line in f:
        lable.append(1 if line.strip() == '1' else 0)
data_df = data_df.join(pd.DataFrame(lable, columns=['label']))
train_dummies = pd.get_dummies(data_df.drop(columns=['native.country']),
                         columns=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex'])
judge_dummies = pd.get_dummies(judge_df.drop(columns=['native.country']),
                               columns=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex'])

# build classifier
xgb = XGBClassifier(n_estimators=111, max_depth=11, min_child_weight=1459507, learning_rate=0.065082245687885)

# train, predict and output
xgb.fit(train_dummies.drop(columns=['fnlwgt', 'label']), train_dummies['label'], sample_weight=train_dummies['fnlwgt'])
predictions = xgb.predict(judge_dummies.drop(columns=['fnlwgt']))
with open("prediction.txt", "w") as f:
    for prediction in predictions:
        print(prediction, file=f)