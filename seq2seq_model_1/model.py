import torch
import torch.nn as nn
import random




class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_value):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_value)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_value)


    def forward(self, x):
        # x shape: (seq_length, N)

        embedding = self.dropout(self.embedding(x))
        #embedding shape: (seq_length, N, embedding_size)


        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell
    



class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_value):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_value)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_value)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        #x shape: (N) -> we want (1, N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        #embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        #outputs shape: (1, N, hidden_size)
        predictions = self.fc(outputs)
        #predictions shape: (1, N, output_size)
        predictions = predictions.squeeze(0)
        #predictions shape: (N, output_size)
        return predictions, hidden, cell
    



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(source.device)

        hidden, cell = self.encoder(source)
        x = target[0] #grab the first token from target

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            #output shape: (N, target_vocab_size)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs
