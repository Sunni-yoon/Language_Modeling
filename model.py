import torch.nn as nn
import torch

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=3, dropout_prob=0.5): 
        super(CharRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_prob) 

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded) 
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output) 
        output = self.fc(output)
        
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=3, dropout_prob=0.5): 
        super(CharLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded) 
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output) 
        output = self.fc(output)
        
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
        

if __name__ == '__main__':
    
    # Test CharRNN
    vocab_size = 50  
    embedding_dim = 64 
    hidden_dim = 256
    num_layers = 3
    dropout_prob = 0.5 
    batch_size = 10 
    seq_length = 20 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CharLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob).to(device)
    x = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long).to(device)
    hidden = model.init_hidden(batch_size, device)
    output, hidden = model(x.to(device), hidden)
    print(f'CharLSTM output shape: {output.shape}')
    print(f'CharLSTM hidden[0] (h_n) shape: {hidden[0].shape}')
    print(f'CharLSTM hidden[1] (c_n) shape: {hidden[1].shape}')
