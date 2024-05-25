import torch
import torch.nn.functional as F
from model import CharRNN, CharLSTM
from dataset import Shakespeare
import numpy as np

def generate(model, seed_characters, temperature, device, char_to_idx, idx_to_char, length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        device: device for computation
        char_to_idx: dictionary to convert characters to indices
        idx_to_char: dictionary to convert indices to characters
        length: length of the generated sequence

    Returns:
        samples: generated characters
    """
    model.eval()
    input_chars = [char_to_idx[char] for char in seed_characters]
    input_tensor = torch.tensor(input_chars, dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1, device)

    samples = seed_characters

    for _ in range(length):
        output, hidden = model(input_tensor, hidden)
        output = output[:, -1, :] / temperature
        probabilities = F.softmax(output, dim=1).squeeze().detach().cpu().numpy()  # detach() 추가
        char_idx = np.random.choice(len(probabilities), p=probabilities)
        char = idx_to_char[char_idx]
        samples += char

        input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)

    return samples

if __name__ == '__main__':
    # Load the trained model with the best validation performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set the model's hyperparameters to match the trained values.
    vocab_size = 62  # Must match the size of the character set used in training
    embedding_dim = 128  # Embedding dimension used in training
    hidden_dim = 128  # Hidden dimension used in training
    num_layers = 2  # Number of LSTM layers used in training
    dropout_prob = 0.5  # Dropout probability used in training

    # Load the dataset to get char_to_idx and idx_to_char
    dataset = Shakespeare('shakespeare_train.txt')
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char

    # Load the trained model
    model_path = 'model_charLSTM.pth'  # Model file path
    model = CharLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob).to(device)
    model.load_state_dict(torch.load(model_path))
    
    seed_characters_list = ["love", "mind", "child", "good", "happy"]
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    for seed_characters in seed_characters_list:
        print(f"Seed(start): {seed_characters}")
        for temp in temperatures:
            generated_text = generate(model, seed_characters, temp, device, char_to_idx, idx_to_char)
            print(f"\nTemperature {temp}:\n{generated_text}\n")

