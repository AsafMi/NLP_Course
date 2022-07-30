import torch
import torch.nn as nn


class TweetNet(nn.Module):
    def __init__(self, model_args, vocab_size):
        super(TweetNet, self).__init__()
        self.lstm_args = model_args.lstm_args
        self.hidden_size = self.lstm_args.hidden_size if not self.lstm_args.bidirectional else self.lstm_args.hidden_size * 2
        self.output_size = model_args.output_size
        self.dropout = model_args.dropout

        # Embedding of dim vocab_size x model_args.lstm_args.input_size
        self.embedding = nn.Embedding(vocab_size+2, model_args.lstm_args.input_size)
        # LSTM
        self.lstm = nn.LSTM(input_size=self.lstm_args.input_size, hidden_size=self.hidden_size,
                            batch_first=self.lstm_args.batch_first, bias=self.lstm_args.bias,
                            num_layers=self.lstm_args.num_layers, bidirectional=self.lstm_args.bidirectional,
                            dropout=self.lstm_args.dropout, proj_size=self.lstm_args.proj_size)
        # Classifier containing dropout, linear layer and sigmoid
        self.linear = nn.Linear(self.hidden_size*2, self.output_size)
        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Softmax()

    def forward(self, input_ids):
        # Embed
        embeds = self.embedding(input_ids)  # (1, seq_length) -> (1, seq_length, input_size)
        embeds = self.dropout(embeds)
        # Run through LSTM and take the final layer's output
        lstm_out, (ht, ct) = self.lstm(embeds)
        # (1, seq_length, input_size) -> (1, max_seq_length, hidden_size)
        # Take the mean of all the output vectors
        seq_embeddings = lstm_out.mean(dim=1)  # (1, max_seq_length, hidden_size) -> (1, hidden_size)

        # Classifier
        logits = self.linear(seq_embeddings)  # (1, hidden_size) -> (1, n_classes)
        logits = logits.float()
        return self.classifier(logits)
