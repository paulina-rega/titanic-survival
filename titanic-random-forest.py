import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel

def does_substring_exists(string_to_check, substring):
    if string_to_check.find(substring) == - 1:
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
    X = X.drop(columns_to_drop, axis = 1)
    X = one_hot_encode(X, columns_to_encode)
    
    X['Cabin'] = X['Cabin'].str.split('').str[1]
    X['Cabin'] = X['Cabin'].fillna('X')
    X = one_hot_encode(X, ['Cabin'])
    
    X['FamSize'] = X['SibSp']+X['Parch']+1
    X['PricePersn'] = X['Fare']/X['FamSize']
    
    
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
    
    
    

    return X


df = pd.read_csv('train.csv')
df = df.dropna(subset = ['Age'])
y = df['Survived']
X = df.drop('Survived', axis=1)
X_test_data = pd.read_csv('test.csv')


columns_to_drop = ['PassengerId', 'Ticket', 'Name']
columns_to_encode = ['Pclass', 'Sex', 'Embarked']

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
'''
# Uncomment this code block to show all features importance

print('\nFeatures importance: ')
for feature in zip(labels, clf.feature_importances_):
    print(\nfeature)
'''

# choosing important fatures for new model
sfm = SelectFromModel(clf, threshold = 0.04)
sfm.fit(x_train, y_train)

# trainning model with only important features
x_important_train = sfm.transform(x_train)
x_important_test = sfm.transform(x_test)
clf_important = RandomForestClassifier(n_estimators=100, random_state = 1, 
                                       n_jobs =-1)
clf_important.fit(x_important_train, y_train)
y_important_pred = clf_important.predict(x_important_test)


# Uncomment this code block to print chosen most important features

print('\nMost important features:')

for feature_list_index in sfm.get_support(indices=True):
    print('\t - ' + labels[feature_list_index])

print('\nAccuracy (chosen parameters with highest importance): {}'.format(
    metrics.accuracy_score(y_important_pred, y_pred)))
