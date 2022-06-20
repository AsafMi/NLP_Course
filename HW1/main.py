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
    # ------------------------------------------------------------------
    # -------------------- Training Data Preprocess --------------------
    # ------------------------------------------------------------------
    # Extracting data, features and labels
    x_train, feature_train, y_train = load_raw_dataset("data\\train.txt")
    # Fitting and transforming OneHotEncoder for the training data
    encoder = OneHotEncoder(handle_unknown="ignore").fit(feature_train)
    feature_train = encoder.transform(feature_train)
    # Encoding the labels
    y_train = np.array(OneHotEncoder(drop='if_binary').fit_transform(y_train).toarray()).ravel()
    print(f"Training data is ready to use\nElapse time: {time.time() - start_time:.02f} sec")
    # ------------------------------------------------------------------
    # -------------------- Validation Data Preprocess ------------------
    # ------------------------------------------------------------------
    # Extracting data, features and labels
    x_val, feature_val, y_val = load_raw_dataset("data\\eval.txt")
    # Transforming the above encoder for the validation data
    feature_val = encoder.transform(feature_val)
    # Encoding the labels
    y_val = np.array(OneHotEncoder(drop='if_binary').fit_transform(y_val).toarray()).ravel()
    print(f"Validation data is ready to use\nElapse time: {time.time() - start_time:.02f} sec")
    # ------------------------------------------------------------------
    # ------------------ Model Definition & Evaluation -----------------
    # ------------------------------------------------------------------
    model = SVC()  # TODO:GRIDSEARCH
    model.fit(feature_train, y_train)  # fitting the model on the training data
    print(f"SVM model was trained\nElapse time: {time.time() - start_time:.02f} sec")
    print(f"Train score: {model.score(feature_train, y_train):.03f}")
    print(f"Eval score: {model.score(feature_val, y_val):.03f}")

    predictions = model.predict(feature_val)  # calculate the predictions on the validation data

    cm = confusion_matrix(y_val, predictions, labels=model.classes_)  # building a confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.show()

    # Saving bad classification for review
    confused_data = [word + f" , {y_val[idx]}\n" for idx, word in enumerate(x_val) if y_val[idx] != predictions[idx]]
    file = open("confusions.txt", "w")
    file.write("Word, true label \n")
    file.writelines(confused_data)
    file.close()

    # create "competitive.txt" for substitution
    # open both files
    with open('data/test.txt', 'r') as firstfile, open('competitive.txt', 'a') as secondfile:
        # read content from test file
        for line in firstfile:
            # append content to second file
            if line == ". .":
                secondfile.write(line + " O\n")
