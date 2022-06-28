import numpy as np
import pandas as pd
import torch as torch
from bs4 import BeautifulSoup
import re
from gensim.models import word2vec
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def DataPreprocess(path):
    df = pd.read_csv(path)
    lReviews = df.review.to_list()
    lSentiment = df.sentiment.to_list()
    lLines = []
    for review in lReviews:
        text   = BeautifulSoup(review, features="html.parser").get_text()                 #-- remove <br> and HTML
        lWords = re.sub("[^a-zA-Z]", " ", text).lower().split() #-- keep lower case letters
        lLines.append(lWords)
    # Train Word2Vec model
    d            = 300
    minWordCount = 40
    contextWin   = 5

    oWord2Vec    = word2vec.Word2Vec(lLines, workers=4, vector_size=d, min_count=minWordCount, window=contextWin)

    x_train = [] #TODO:CHANGE VAR NAMES
    for line in lLines:
        vec = np.zeros(d).reshape((1, d))
        count = 0
        for word in line:
            vec += oWord2Vec.wv[word].reshape((1, d))
            count += 1
        if count != 0:
            vec /= count
        x_train.append(vec)

    x_train = np.concatenate(x_train)
    return scale(x_train)

x_train = DataPreprocess(r"./data/IMDB_train.csv")
x_val = DataPreprocess(r"./data/IMDB_test.csv.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# We create a FC regression network, with 2 layers.
class RegressioNet(nn.Module):
   def __init__(self):
       super(RegressioNet, self).__init__()
       self.hidden_dim = 10
       self.layer_1 = torch.nn.Linear(1, self.hidden_dim)
       self.layer_2 = torch.nn.Linear(self.hidden_dim, 1)
       self.activation = F.relu

   def forward(self, x):
       x = self.layer_1(x)        # x.size() -> [batch_size, self.hidden_dim]
       x = self.activation(x)     # x.size() -> [batch_size, self.hidden_dim]
       x = self.layer_2(x)        # x.size() -> [batch_size, 1]
       return x

net = RegressioNet()

# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.BCELoss()

batch_size = 20

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(train_y).float())
valid_data = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(val_y).float())
#test_data = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
#test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# Define training params
epochs = 1

counter = 0
print_every = 100
clip = 1000 # gradient clipping

# move model to GPU, if available
net = net.float()
net.to(device)

net.train()
# train for some number of epochs

for e in range(epochs):
    # batch loop
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

        # calculate the loss and perform backprop
        loss = loss_func(predictions.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_losses = []
            net.eval()
            print_flag = True
            for inputs, labels in valid_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                if print_flag:
                    inputs, labels = zip(*sorted(zip(inputs.numpy(), labels.numpy())))
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
                val_loss = loss_func(val_predictions.squeeze(), labels.float())

                val_losses.append(val_loss.item())
                if print_flag:
                    print_flag = False
                    # plot and show learning process
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.cla()
                    ax.scatter(inputs.cpu().data.numpy(), labels.cpu().data.numpy())
                    ax.plot(inputs.cpu().data.numpy(), val_predictions.cpu().data.numpy(), 'r-', lw=2)
                    ax.text(0.5, 0, 'Loss=%.4f' % np.mean(val_losses), fontdict={'size': 10, 'color': 'red'})
                    plt.pause(0.1)
                    ax.clear()

            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
plt.show()














