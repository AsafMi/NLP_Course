import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from gensim.models import word2vec
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
np.random.seed(23)
torch.manual_seed(23)
def DataPreprocess(path):
    df = pd.read_csv(path)
    lReviews = df.review.to_list()
    lSentiment = df.sentiment.to_list()
    y_train = [1. if s == "positive" else 0. for s in lSentiment]
    y_train = np.asarray(y_train, dtype=np.float32)
    lLines = []
    for review in lReviews:
        text   = BeautifulSoup(review, features="html.parser").get_text()                 #-- remove <br> and HTML
        lWords = re.sub("[^a-zA-Z]", " ", text).lower().split() #-- keep lower case letters
        lLines.append(lWords)
    # Train Word2Vec model
    d            = 200
    minWordCount = 10
    contextWin   = 5

    oWord2Vec    = word2vec.Word2Vec(lLines, workers=4, vector_size=d, min_count=minWordCount, window=contextWin)

    x_train = []
    for line in lLines:
        vec = np.zeros(d).reshape((1, d))
        count = 0
        for word in line:
            try:
                vec += oWord2Vec.wv[word].reshape((1, d))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        x_train.append(vec)

    x_train = np.concatenate(x_train)

    return scale(x_train), y_train

x_train, y_train = DataPreprocess(r"./data/IMDB_train.csv")
x_val, y_val = DataPreprocess(r"./data/IMDB_test.csv")
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# We create a FC regression network, with 2 layers.
class RegressioNet(nn.Module):
   def __init__(self):
       super(RegressioNet, self).__init__()
       self.hidden_dim = 5
       self.layer_1 = torch.nn.Linear(200, self.hidden_dim)
       self.layer_2 = torch.nn.Linear(self.hidden_dim, 1)
       self.activation = F.relu
       self.drop = torch.nn.Dropout(p=0.5, inplace=False)

   def forward(self, x):
       x = self.layer_1(x)        # x.size() -> [batch_size, self.hidden_dim]
       x = self.activation(x)     # x.size() -> [batch_size, self.hidden_dim]
       x = self.drop(x)
       x = self.layer_2(x)        # x.size() -> [batch_size, 1]
       return x

net = RegressioNet()

# Define Optimizer and Loss Function
#optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_func = torch.nn.BCEWithLogitsLoss()

batch_size = 20

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
valid_data = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
#test_data = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
#test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# Define training params
epochs = 10

counter = 0
print_every = 100

# move model to GPU, if available
net = net.float()
net.to(device)

net.train()
# train for some number of epochs
loss_vals=[]
train_loss=[]


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag.squeeze() == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc
train_acc_list = []
val_acc_list = []
for e in range(epochs):
    # batch loop
    train_acc=0
    train_acc_list_temp = []
    for inputs, labels in train_loader:
        counter += 1

        # if training on gpu
        inputs, labels = inputs.to(device), labels.to(device)

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        # x.size() -> [batch_size]
        batch_size = inputs.size(0)
        # IMPORTANT - change the dimensions of x before it enters the NN, batch size must always be first
        x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
        x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
        predictions = net(x)
        train_acc = binary_acc(predictions, labels)
        train_acc_list_temp.append(train_acc)
        # calculate the loss and perform backprop
        loss = loss_func(predictions.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_losses = []
            val_acc_list_temp = []
            val_acc=0
            net.eval()
            print_flag = True
            for inputs, labels in valid_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                if print_flag:
                    inputs, labels = zip(*sorted(zip(inputs.numpy(), labels.numpy()), key=lambda x: x[1]))
                    inputs = torch.from_numpy(np.asarray(inputs))
                    labels = torch.from_numpy(np.asarray(labels))
                inputs, labels = inputs.to(device), labels.to(device)

                # get the output from the model
                # x.size() -> [batch_size]
                batch_size = inputs.size(0)
                # IMPORTANT - change the dimensions of x before it enters the NN, batch size must always be first
                x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
                x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
                val_predictions = net(x)
                val_acc = binary_acc(val_predictions, labels)
                val_loss = loss_func(val_predictions.squeeze(), labels.float())

                val_losses.append(val_loss.item())
                val_acc_list_temp.append(val_acc)
            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)),
                  "Acc: {:.6f}...".format(np.mean(train_acc_list_temp)),
                  "Val Acc: {:.6f}".format(np.mean(val_acc_list_temp)))
            loss_vals.append(np.mean(val_losses))
            train_loss.append(loss.item())
            val_acc_list.append(np.mean(val_acc_list_temp))
    train_acc_list.append(np.mean(train_acc_list_temp))
plt.plot(np.linspace(1, epochs, len(loss_vals)).astype(float), loss_vals, label= "val Loss")
plt.plot(np.linspace(1, epochs, len(train_loss)).astype(float), train_loss, label= "train Loss")
plt.legend()
plt.show()
plt.plot(np.linspace(1, epochs, len(train_acc_list)).astype(float), train_acc_list, label= "train acc")
plt.plot(np.linspace(1, epochs, len(val_acc_list)).astype(float), val_acc_list, label= "val acc")
plt.legend()
plt.show()












