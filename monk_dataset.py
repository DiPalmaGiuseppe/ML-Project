import pandas as pd
from sklearn.preprocessing import StandardScaler

    
def monk_loader(id, normalize = False):
    train = pd.read_csv(f"monk_dataset/monks-{id}.train", sep = ' ', header = None, usecols=range(1, 8))
    test = pd.read_csv(f"monk_dataset/monks-{id}.test", sep = ' ', header = None, usecols=range(1, 8))

    y_train = train.iloc[:, 0]
    X_train = train.iloc[:, 1:]

    y_test = test.iloc[:, 0]
    X_test = test.iloc[:, 1:]
    
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test