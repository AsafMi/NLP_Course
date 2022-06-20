from dataloading import *
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

start_time = time.time()
plotConfusionMatrix = False
saveConfusions = False

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # -------------------- Training Data Preprocess --------------------
    # ------------------------------------------------------------------
    # Extracting data, features and labels
    # x_train, feature_train, y_train = load_raw_dataset("data\\train.txt")
    # # Fitting and transforming OneHotEncoder for the training data
    # encoder = OneHotEncoder(handle_unknown="ignore").fit(feature_train)
    # feature_train = encoder.transform(feature_train)
    # print(f"Training data is ready to use\nElapse time: {time.time() - start_time:.02f} sec")
    # # ------------------------------------------------------------------
    # # -------------------- Validation Data Preprocess ------------------
    # # ------------------------------------------------------------------
    # # Extracting data, features and labels
    # x_val, feature_val, y_val = load_raw_dataset("data\\eval.txt")
    # # Transforming the above encoder for the validation data
    # feature_val = encoder.transform(feature_val)
    # print(f"Validation data is ready to use\nElapse time: {time.time() - start_time:.02f} sec")
    # # ------------------------------------------------------------------
    # # ------------------ Model Definition & Evaluation -----------------
    # # ------------------------------------------------------------------
    # model = SVC()
    # model.fit(feature_train, y_train)  # fitting the model on the training data
    # print(f"SVM model was trained\nElapsed time: {time.time() - start_time:.02f} sec")
    # print(f"Train score: {model.score(feature_train, y_train):.03f}")
    # print(f"Eval score: {model.score(feature_val, y_val):.03f}")
    #
    # predictions = model.predict(feature_val)  # calculate the predictions on the validation data
    # if plotConfusionMatrix:
    #     cm = confusion_matrix(y_val, predictions, labels=model.classes_)  # building a confusion matrix
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                                   display_labels=model.classes_)
    #     disp.plot()
    #     plt.show()
    #
    # if saveConfusions: # Saving bad classification for review
    #     confused_data = [word + f" , {y_val[idx]}\n" for idx, word in enumerate(x_val) if y_val[idx] != predictions[idx]]
    #     file = open("confusions.txt", "w")
    #     file.write("Word, true label \n")
    #     file.writelines(confused_data)
    #     file.close()
    # ------------------------------------------------------------------
    # ------------ Create "competitive.txt" for substitution -----------
    # ------------------------------------------------------------------
    # load "train.txt" and "eval.txt"
    x_train, feature_train, y_train = load_raw_dataset("data\\train.txt")
    # x_val, feature_val, y_val = load_raw_dataset("data\\eval.txt")
    # concatenate train and eval
    # x_train.extend(x_val)
    # feature_train.extend(feature_val)
    # y_train.extend(y_val)
    # One-Hot-Encoding of the data
    x_encoder = OneHotEncoder(handle_unknown="ignore").fit(feature_train)
    feature_train = x_encoder.transform(feature_train)
    # train on a bigger dataset (train + eval)
    model = SVC()
    model.fit(feature_train, y_train)
    # open both files
    open('competitive.txt', 'w').close()
    with open('data/train.txt', 'r') as firstfile, open('competitive.txt', 'a') as secondfile:
        # read content from test file
        for line in tqdm(firstfile):
            data = line.strip().split(sep=" ")
            if len(data) < 2:
                secondfile.write(line)
                continue
            x = data[0]
            feature = data[1]
            # append content to second file
            if x in [".", ",", "-"]:
                secondfile.write(x + " " + feature + " " + "O\n")
                continue
            else:
                capital_feature = x.isupper()
                x_test = [x, feature, capital_feature]
                x_test = x_encoder.transform([x_test])
                y_test = model.predict(x_test)
                secondfile.write(x + " " + feature + " " + f"{y_test[0]}\n")
