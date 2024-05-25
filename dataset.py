import torch
from torch.utils.data import Dataset


class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        # Load the input file
        with open(input_file, 'r') as file:
            text = file.read()

        # Create character dictionary
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

        # Convert all characters in the text to their corresponding indices
        self.text_indices = [self.char_to_idx[char] for char in text]

        # Define the sequence length
        self.sequence_length = 30

        # Create input and target sequences
        self.data = []
        for i in range(len(self.text_indices) - self.sequence_length):
            input_seq = self.text_indices[i:i + self.sequence_length]
            target_seq = self.text_indices[i + 1:i + self.sequence_length + 1]
            self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)
        return input_seq, target_seq

if __name__ == '__main__':
    
    # Test the implementation
    dataset = Shakespeare('shakespeare_train.txt')
    
    # Check the length of the dataset
    print("Length of dataset:", len(dataset))
    
    vocab_size = len(dataset.chars)
    print("Vocab size:", vocab_size)

    # Get a sample item from the dataset
    sample_input, sample_target = dataset[0]
    print("Sample input:", sample_input)
    print("Sample input shape:", sample_input.shape)
    print("Sample target:", sample_target)
    print("Sample target shape:", sample_target.shape)

    # Print the input and target sequences as characters
    input_text = ''.join([dataset.idx_to_char[idx.item()] for idx in sample_input])
    target_text = ''.join([dataset.idx_to_char[idx.item()] for idx in sample_target])
    print(f'Input text: {input_text}')
    print(f'Target text: {target_text}')