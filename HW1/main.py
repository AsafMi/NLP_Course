from dataloading import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x_train, feature_train, y_train = load_raw_dataset("data\\train.txt")
    capital_feature = convert_raw_to_features(x_train)
    feature_train = [list(x) for x in zip(feature_train, capital_feature)]
    feature_train = OneHotEncoder().fit_transform(feature_train)
    y_train = np.array(OneHotEncoder(drop='if_binary').fit_transform(y_train).toarray()).ravel()

    x_val, feature_val, y_val = load_raw_dataset("data\\eval.txt")
    capital_feature = convert_raw_to_features(x_val)
    feature_val = [list(x) for x in zip(feature_val, capital_feature)]
    feature_val = OneHotEncoder().fit_transform(feature_val)
    y_val = np.array(OneHotEncoder(drop='if_binary').fit_transform(y_val).toarray()).ravel()

    model = SVC()

    model.fit(feature_train, y_train)

    print(f"Train score: {model.score(feature_train, y_train)}")
    print(f"Eval score: {model.score(feature_val, y_val)}")

    predictions = model.predict(feature_val)
    cm = confusion_matrix(y_val, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.show()
