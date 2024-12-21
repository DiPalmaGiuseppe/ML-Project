import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def monk_loader(id):
    # Load the training and test data
    train = pd.read_csv(f"monk_dataset/monks-{id}.train", sep=' ', header=None, usecols=range(1, 8))
    test = pd.read_csv(f"monk_dataset/monks-{id}.test", sep=' ', header=None, usecols=range(1, 8))

    y_train = train.iloc[:, 0]
    X_train = train.iloc[:, 1:]

    y_test = test.iloc[:, 0]
    X_test = test.iloc[:, 1:]

    encoder = OneHotEncoder(sparse_output = False)

    X_train = pd.DataFrame(encoder.fit_transform(X_train)).astype(int)
    X_test = pd.DataFrame(encoder.transform(X_test)).astype(int)

    return (
        X_train,  # One-hot encoded training features as DataFrame
        y_train,             # Training labels
        X_test,   # One-hot encoded test features as DataFrame
        y_test               # Test labels
    )
