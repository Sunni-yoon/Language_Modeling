import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataset import Shakespeare
from model import CharRNN, CharLSTM
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """
    model.train()
    total_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0), device)
        output, hidden = model(inputs, hidden)
        output = output.view(-1, output.size(2))
        targets = targets.view(-1)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    trn_loss = total_loss / len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0), device)
            output, hidden = model(inputs, hidden)
            output = output.view(-1, output.size(2))
            targets = targets.view(-1)
            loss = criterion(output, targets)
            total_loss += loss.item()
    val_loss = total_loss / len(val_loader)
    return val_loss

def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        2) model
        3) optimizer
        4) cost function: use torch.nn.CrossEntropyLoss

    """
    # Hyperparameters
    batch_size = 128
    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2
    num_epochs = 10
    learning_rate = 0.0005
    validation_split = 0.2

    # Load dataset
    dataset = Shakespeare('shakespeare_train.txt')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device : {device}')  
    vocab_size = len(dataset.chars)

    model_rnn = CharRNN(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    model_lstm = CharLSTM(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=learning_rate)
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=learning_rate)

    # Lists to store loss values
    trn_loss_list_rnn, val_loss_list_rnn = [], []
    trn_loss_list_lstm, val_loss_list_lstm = [], []

    # Train the RNN model
    print("Training CharRNN...")
    for epoch in range(num_epochs):
        trn_loss = train(model_rnn, train_loader, device, criterion, optimizer_rnn)
        val_loss = validate(model_rnn, val_loader, device, criterion)
        trn_loss_list_rnn.append(trn_loss)
        val_loss_list_rnn.append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save the RNN model
    torch.save(model_rnn.state_dict(), 'model_charRNN.pth')

    # Train the LSTM model
    print("Training CharLSTM...")
    for epoch in range(num_epochs):
        trn_loss = train(model_lstm, train_loader, device, criterion, optimizer_lstm)
        val_loss = validate(model_lstm, val_loader, device, criterion)
        trn_loss_list_lstm.append(trn_loss)
        val_loss_list_lstm.append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the LSTM model
    torch.save(model_lstm.state_dict(), 'model_charLSTM.pth')

    # Plot the training and validation loss RNN & LSTM 
    plt.figure(figsize=(10, 4))
    plt.plot(trn_loss_list_rnn, label='Train Loss RNN')
    plt.plot(val_loss_list_rnn, label='Val Loss RNN')
    plt.plot(trn_loss_list_lstm, label='Train Loss LSTM')
    plt.plot(val_loss_list_lstm, label='Val Loss LSTM')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('RNN & LSTM Training and Validation Loss')
    plt.savefig('rnn_lstm_training_validation_loss.png')
    plt.show()
    
    # Plot the RNN training and validation loss separately
    plt.figure(figsize=(10, 4))
    plt.plot(trn_loss_list_rnn, label='Train Loss RNN', color='blue', marker='o')
    plt.plot(val_loss_list_rnn, label='Val Loss RNN', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('RNN Training and Validation Loss')
    plt.savefig('rnn_training_validation_loss.png')
    plt.show()

    # Plot the LSTM training and validation loss separately
    plt.figure(figsize=(10, 4))
    plt.plot(trn_loss_list_lstm, label='Train Loss LSTM', color='blue', marker='o')
    plt.plot(val_loss_list_lstm, label='Val Loss LSTM', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('LSTM Training and Validation Loss')
    plt.savefig('lstm_training_validation_loss.png')
    plt.show()


if __name__ == '__main__':
    main()
