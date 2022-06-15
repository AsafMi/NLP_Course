from dataloading import *
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    x_train, feature_train, y_train = load_raw_dataset("data\\train.txt")
    capital_feature = convert_raw_to_features(x_train)
    feature_train = [list(x) for x in zip(feature_train, capital_feature)]
    feature_train = OneHotEncoder().fit_transform(feature_train)
    y_train = OneHotEncoder(drop='if_binary').fit_transform(y_train)
