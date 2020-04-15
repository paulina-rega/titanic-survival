import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel

def does_substring_exists(string_to_check, substring):
    if string_to_check.find(substring) != - 1:
        return True
    else:
        return False


def one_hot_encode(df, columns_to_encode):
    for col in columns_to_encode:
        one_hot = pd.get_dummies(df[col], prefix=col)
        df = df.drop(col, axis=1)
        df = df.join(one_hot)
    return df


def preprocess_feature_matrix(X, columns_to_drop, columns_to_encode):
    X = X.fillna(0)
    X = X.drop(columns_to_drop, axis = 1)
    X = one_hot_encode(X, columns_to_encode)
    
    X['Cabin'] = X['Cabin'].str.split('').str[1]
    X['Cabin'] = X['Cabin'].fillna('X')
    X = one_hot_encode(X, ['Cabin'])
    
    X['FamSize'] = X['SibSp']+X['Parch']+1
    X['PricePersn'] = X['Fare']/X['FamSize']
    return X


df = pd.read_csv('train.csv')

title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']

titles = pd.Series([])
for index, row in df.iterrows():
    is_changed = 0
    for title in title_list:
        if does_substring_exists(row['Name'], title):
            titles = titles.append(pd.Series([title]))
            break
        
titles = titles.reset_index(drop=True).to_frame(name="Title")
df = pd.concat([df, titles], axis=1)
df = df.drop('Name', axis = 1)


mean_age = df.groupby('Title').mean()['Age']


for index, row in df.iterrows():
    df.loc[~(df['Age'] > 0), ['Age']] = mean_age[row['Title']]


X_test_data = pd.read_csv('test.csv')

titles = pd.Series([])
for index, row in X_test_data.iterrows():
    is_changed = 0
    for title in title_list:
        if does_substring_exists(row['Name'], title):
            titles = titles.append(pd.Series([title]))
            break
        
titles = titles.reset_index(drop=True).to_frame(name="Title")
X_test_data = pd.concat([X_test_data, titles], axis=1)
X_test_data = X_test_data.drop('Name', axis = 1)


for index, row in X_test_data.iterrows():
    X_test_data.loc[~(X_test_data['Age'] > 0), ['Age']] = mean_age[row['Title']]


y = df['Survived']
X = df.drop('Survived', axis=1)


columns_to_drop = ['PassengerId', 'Ticket']
columns_to_encode = ['Pclass', 'Sex', 'Embarked', 'Title']

X = preprocess_feature_matrix(X, columns_to_drop, columns_to_encode)
X_test_data = preprocess_feature_matrix(X_test_data, columns_to_drop, 
                                        columns_to_encode)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

clf = RandomForestClassifier(n_estimators=100, random_state = 1, n_jobs =-1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print('Accuracy (all given parameters considered): {}'.format(
    metrics.accuracy_score(y_test, y_pred)))


labels = list(X.columns)


neccessary_columns = list(set(list(X.columns)+list(X_test_data.columns)))

for column in neccessary_columns:
    if column not in list(X_test_data.columns):
        X_test_data[column] = X_test_data.apply(lambda x: 0, axis=1)
        
x_train = x_train[X_test_data.columns]
x_test = x_test[X_test_data.columns]

# choosing important fatures for new model
sfm = SelectFromModel(clf, threshold = 0.03)
sfm.fit(x_train, y_train)

# trainning model with only important features
x_important_train = sfm.transform(x_train)
x_important_test = sfm.transform(x_test)
X_important_test_data = sfm.transform(X_test_data)
clf_important = RandomForestClassifier(n_estimators=100, random_state = 1, 
                                       n_jobs =-1)
clf_important.fit(x_important_train, y_train)
y_important_pred = clf_important.predict(x_important_test)

y_important_pred_test = clf_important.predict(X_important_test_data)

print('\nMost important features:')

for feature_list_index in sfm.get_support(indices=True):
    print('\t - ' + labels[feature_list_index])

print('\nAccuracy (chosen parameters with highest importance): {}'.format(
    metrics.accuracy_score(y_important_pred, y_pred)))

# exporting CSV file with prediction

prediction_table = pd.read_csv('test.csv')['PassengerId']
y_important_pred_test = pd.Series(y_important_pred_test)
prediction_table = prediction_table.to_frame().join(y_important_pred_test.to_frame())
prediction_table = prediction_table.rename(columns={0: 'Survived'})
prediction_table.to_csv('prediction.csv', header='survival', index=False)


        
