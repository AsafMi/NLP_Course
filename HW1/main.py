from dataloading import *

if __name__ == '__main__':
    x_train, feature_train, y_train = load_raw_dataset("data\\train.txt") #TODO: feature_train >> one hot encoding
    capital_feature = convert_raw_to_features(x_train) #TODO: capital_feature boolian to 1,0
    feature_train = [list(x) for x in zip(feature_train, capital_feature)]
