import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MinMaxTransform:
    def __init__(self):
        # Initialize the MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit(self, data):
        # Fit the scaler on the data
        self.scaler.fit(data)

    def __call__(self, sample):
        # Apply the transformation to the sample
        return self.scaler.transform(sample.reshape(1, -1)).flatten()

class BinaryClassificationDataset(Dataset):
    def __init__(self, attack_file, benign_file, transform=None):
        # Load data from files
        self.attack_data = pd.read_csv(attack_file)
        self.benign_data = pd.read_csv(benign_file)
        
        # Combine the datasets
        self.data = pd.concat([self.attack_data, self.benign_data], ignore_index=True)
        
        # Optionally shuffle the data if the model training requires randomness, run transforms
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        # Return the length of the data
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to list just incase
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Assuming the last column is the label
        sample = self.data.iloc[idx, :-1].values.astype('float')
        label = self.data.iloc[idx, -1].astype('int')

        # Transform the samples if present
        if self.transform:
            sample = self.transform(sample)

        # Return the samples and labels as torch tensors
        return torch.tensor(sample), torch.tensor(label)


def main():
    # Define file paths
    train_attack = '/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/train_attack.csv'
    train_benign = '/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/train_benign.csv'
    test_attack = '/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/test_attack.csv'
    test_benign = '/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/test_benign.csv'

    # Load train data for calculating normalization
    temp_attack_data = pd.read_csv(train_attack)
    temp_benign_data = pd.read_csv(train_benign)
    temp_train_data = pd.concat([temp_attack_data, temp_benign_data], ignore_index=True)

    # Initialize transform based on the entire train dataset
    min_max_transform = MinMaxTransform()
    min_max_transform.fit(temp_train_data.iloc[:, :-1])

    # Delete the temp data used to find the transform
    del temp_attack_data, temp_benign_data, temp_train_data

    # Create dataset instances
    train_dataset = BinaryClassificationDataset(train_attack, train_benign, transform=min_max_transform)
    test_dataset = BinaryClassificationDataset(test_attack, test_benign, transform=min_max_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)

    # Example: Iterate over the train loader
    for data, labels in train_loader:
        print(data.shape, labels.shape) 

# Run main function
if __name__ == "__main__":
    main()