from dataloading import *
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

if __name__ == '__main__':
    x_train, feature_train, y_train = load_raw_dataset("data\\train.txt")
    capital_feature = [word[0].isupper() for word in x_train]
    abc_feature = [word[0].isalpha() for word in x_train]
    day_feature = InListFeature(x_train)
    feature_train = [list(x) for x in zip(x_train, feature_train, capital_feature, abc_feature, day_feature)]
    encoder = OneHotEncoder(handle_unknown="ignore").fit(feature_train)
    feature_train = encoder.transform(feature_train)
    y_train = np.array(OneHotEncoder(drop='if_binary').fit_transform(y_train).toarray()).ravel()
    print(f"Training data is ready to use\nElapse time: {time.time() - start_time:.02f} sec")

    x_val, feature_val, y_val = load_raw_dataset("data\\eval.txt")
    capital_feature = [word[0].isupper() for word in x_val]
    abc_feature = [word[0].isalpha() for word in x_val]
    day_feature = InListFeature(x_val)
    feature_val = [list(x) for x in zip(x_val, feature_val, capital_feature, abc_feature, day_feature)]
    feature_val = encoder.transform(feature_val)
    y_val = np.array(OneHotEncoder(drop='if_binary').fit_transform(y_val).toarray()).ravel()
    print(f"Validation data is ready to use\nElapse time: {time.time() - start_time:.02f} sec")

    model = SVC()

    model.fit(feature_train, y_train)
    print(f"SVM model was trained\nElapse time: {time.time() - start_time:.02f} sec")

    print(f"Train score: {model.score(feature_train, y_train):.03f}")
    print(f"Eval score: {model.score(feature_val, y_val):.03f}")

    predictions = model.predict(feature_val)

    cm = confusion_matrix(y_val, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.show()

    confused_data = [word + f" , {y_val[idx]}\n" for idx, word in enumerate(x_val) if y_val[idx] != predictions[idx]]
    file1 = open("confusions.txt", "w")
    file1.write("Word, true label \n")
    file1.writelines(confused_data)
    file1.close()  # to change file access modes



    a = 1
