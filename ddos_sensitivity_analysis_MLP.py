import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import sys

# Define constants
BATCH_SIZE = 1000

# Determine device
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)


class MinMaxTransform:
    def __init__(self):
        # Initialize the MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit(self, data):
        # Fit the scaler on the data
        self.scaler.fit(data)

    def __call__(self, sample):
        # Make sure that we run the transform with the features loaded
        feature_names = self.scaler.feature_names_in_
        sample_df = pd.DataFrame(sample.reshape(1, -1), columns=feature_names)
        transformed_sample = self.scaler.transform(sample_df)
        return transformed_sample.flatten()


class mlpDataset(Dataset):
    def __init__(self, attack_file, benign_file, transform=None, sample_type=None):
        # If BENIGN was passed, just load that set
        if sample_type == "BENIGN":
            self.data = pd.read_csv(benign_file)
            self.data.iloc[:, -1] = 0
        else:
            self.data = pd.read_csv(attack_file)
            # If anything else was passed, only extract samples of that type
            self.data = self.data[self.data[" Label"] == sample_type]
            self.data.iloc[:, -1] = 1

        # Store transform
        self.transform = transform

    def __len__(self):
        # Return the length of the data
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to list just incase
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Grab last column as label
        sample = self.data.iloc[idx, :-1].values.astype("float")
        label = self.data.iloc[idx, -1]

        # Transform the samples if present
        if self.transform:
            sample = self.transform(sample)

        # Return the samples and labels as torch tensors
        return torch.tensor(sample), torch.tensor(label)


class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        # 79 input features
        self.layer1 = nn.Linear(79, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x


def test(model, test_loader, criterion):
    # Set the model to evaluation mode, save variables for tracking metrics
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient updates, loop over test loader
    with torch.no_grad():
        for data, labels in test_loader:
            # Ensure labels are in the correct shape
            data, labels = data.float().to(device), labels.float().unsqueeze(1).to(device)

            # Forward pass on batch
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Calculate loss and accuracy, store correct predictions
            running_loss += loss.item()
            predicted = outputs.round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Return the average loss and accuracy, in addition to total and correct
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, correct, total


def main():
    # Define file paths
    train_attack = (
        "/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/train_attack.csv"
    )
    train_benign = (
        "/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/train_benign.csv"
    )
    test_attack = (
        "/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/test_attack.csv"
    )
    test_benign = (
        "/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/test_benign.csv"
    )

    # Load train data for calculating normalization
    temp_attack_data = pd.read_csv(train_attack)
    temp_benign_data = pd.read_csv(train_benign)
    temp_train_data = pd.concat([temp_attack_data, temp_benign_data], ignore_index=True)

    # Initialize transform based on the entire train dataset
    min_max_transform = MinMaxTransform()
    min_max_transform.fit(temp_train_data.iloc[:, :-1])

    # Delete the temp data used to find the transform
    del temp_attack_data, temp_benign_data, temp_train_data

    # Create dataset instance
    test_dataset = mlpDataset(
        test_attack, test_benign, transform=min_max_transform, sample_type=sys.argv[1]
    )

    # Create DataLoader
    sensitivity_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )

    # Setup model, loss function, and optimizer, load model
    model = mlp().to(device)
    state_dict = torch.load('/s/bach/b/class/cs535/cs535b/ddos-classification/MLP_model', map_location=torch.device("cpu"))
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    criterion = nn.BCELoss()

    # Test the model and print loss and accuracy, as well as correct guesses, total guesses, and the tested sample type
    test_loss, test_accuracy, test_correct, test_total = test(model, sensitivity_loader, criterion)
    print(
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test Correct: {test_correct}, Test Total: {test_total}, Sample type: {sys.argv[1]}"
    )

# Run main function
if __name__ == "__main__":
    print("Starting sensitivity analysis...")
    main()
