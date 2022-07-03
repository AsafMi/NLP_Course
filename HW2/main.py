import time
from gensim.models import word2vec
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from functions import *

# Using random seed as demanded
np.random.seed(23)
torch.manual_seed(23)
# Loading training data + preprocess
lLines_train, y_train = dataPreprocess(r"./data/IMDB_train.csv")
# Train Word2Vec model
d = 200
minWordCount = 10
contextWin = 5

oWord2Vec = word2vec.Word2Vec(lLines_train, workers=4, vector_size=d, min_count=minWordCount, window=contextWin)

x_train = line2vec(lLines_train, oWord2Vec)

# Loading validation data + preprocess
lLines_val, y_val = dataPreprocess(r"./data/IMDB_test.csv")
x_val = line2vec(lLines_val, oWord2Vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# We create a FC regression network, with 2 layers.
class RegressioNet(nn.Module):
    def __init__(self):
        super(RegressioNet, self).__init__()
        self.layer_1 = torch.nn.Linear(d, 100)
        self.layer_2 = torch.nn.Linear(100, 50)
        self.layer_3 = torch.nn.Linear(50, 1)
        self.activation = F.relu
        self.drop = torch.nn.Dropout(p=0.1, inplace=False)

    def forward(self, x):
        x = self.layer_1(x)  # x.size() -> [batch_size, self.hidden_dim]
        x = self.activation(x)  # x.size() -> [batch_size, self.hidden_dim]
        x = self.drop(x)
        x = self.layer_2(x)  # x.size() -> [batch_size, 1]
        x = self.activation(x)  # x.size() -> [batch_size, self.hidden_dim]
        x = self.drop(x)
        x = self.layer_3(x)  # x.size() -> [batch_size, 1]
        return x


net = RegressioNet()

# Define Loss, Dataset, DataLoader, Optimizer and scheduler Functions

loss_func = torch.nn.BCEWithLogitsLoss()
batch_size = 30

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
valid_data = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

# Define training params
epochs = 50
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)

counter = 0
print_every = 100

# move model to GPU, if available
net = net.float()
net.to(device)

net.train()
# train for some number of epochs
loss_vals = []
train_loss = []

train_acc_list = []
val_acc_list = []
for e in range(epochs):
    # batch loop
    train_acc = 0
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
        optimizer.step()
        scheduler.step()
        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_losses = []
            val_acc_list_temp = []
            val_acc = 0
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
                  "Loss: {:.2f}...".format(loss.item()),
                  "Val Loss: {:.2f}".format(np.mean(val_losses)),
                  "Acc: {:.2f}...".format(np.mean(train_acc_list_temp)),
                  "Val Acc: {:.2f}".format(np.mean(val_acc_list_temp)))
            loss_vals.append(np.mean(val_losses))
            train_loss.append(loss.item())
            val_acc_list.append(np.mean(val_acc_list_temp))
    train_acc_list.append(np.mean(train_acc_list_temp))

fig = plt.figure()
plt.plot(np.linspace(1, epochs, len(loss_vals)).astype(float), loss_vals, label="val Loss")
plt.plot(np.linspace(1, epochs, len(train_loss)).astype(float), train_loss, label="train Loss")
plt.legend()
fig.suptitle('Loss - epochs', fontsize=20)
plt.xlabel('epochs', fontsize=18)
plt.ylabel('Loss', fontsize=16)
fig.savefig(r"./Loss.jpg")


fig = plt.figure()
plt.plot(np.linspace(1, epochs, len(train_acc_list)).astype(float), train_acc_list, label="train acc")
plt.plot(np.linspace(1, epochs, len(val_acc_list)).astype(float), val_acc_list, label="val acc")
plt.legend()
fig.suptitle('Accuracy - epochs', fontsize=20)
plt.xlabel('epochs', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
fig.savefig(r"./Accuracy.jpg")

