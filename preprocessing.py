import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocessing_data(dataset, test_size=.25):
    if dataset in ['breast_cancer_diagnostic', 'breast_cancer_wisconsin']:
        df = pd.read_csv('datasets/' + dataset + '.csv', header = None, index_col = 0)
    elif dataset == 'parkinsons':
        df = pd.read_csv('datasets/' + dataset + '.csv', header = 0, index_col = 0)
    elif dataset == 'wholesale':
        df = pd.read_csv('datasets/' + dataset + '.csv', header = 0)
    else:
        df = pd.read_csv('datasets/' + dataset + '.csv', header = None)

    if dataset in ['breast_cancer_diagnostic', 'breast_cancer_wisconsin', 'cleveland']:
        df = df[df.apply(lambda col: col != str('?'))]

    df.dropna(axis = 0, inplace = True)
    df.drop_duplicates(inplace = True)

    y = df[df.columns[-1]]
    x = df[df.columns[:-1]]

    x = x.values
    y = y.values
    y_new = y.copy()
    classes, counts = np.unique(y, return_counts = True)

    if dataset == 'cleveland':
        y_new[y == 0] = -1
        y_new[y != 0] = 1
    elif dataset == 'breast_cancer_diagnostic':
        y_new[y == 'B'] = -1
        y_new[y == 'M'] = 1
    elif dataset == 'sonar':
        y_new[y == 'M'] = -1
        y_new[y == 'R'] = 1
    elif dataset == 'ionosphere':
        y_new[y == 'b'] = -1
        y_new[y == 'g'] = 1
    else:
        y_new[y == min(classes)] = -1
        y_new[y == max(classes)] = 1

    y_new = y_new.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size = test_size, shuffle = True, random_state = 0, stratify = y_new)

    #Scaling of the data
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test