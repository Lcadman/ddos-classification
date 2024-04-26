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


class convNetDataset(Dataset):
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

        # Get the features and label
        features = self.data.iloc[idx, :-1].values.astype("float")
        label = self.data.iloc[idx, -1]

        # Apply transform if available
        if self.transform:
            features = self.transform(features)

        # Reshape to add channel dimension
        features = features.reshape(1, -1)

        # Convert to tensors, ensure float 32 type
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return features_tensor, label_tensor


class convNet(nn.Module):
    def __init__(self, input_features=80):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 40, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(40, 80, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(3120, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        num_features = x.shape[1]
        if not hasattr(self, 'fc1'):
            # Initialize the fc1 layer dynamically
            self.fc1 = nn.Linear(num_features, 100).to(x.device)
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
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
    test_dataset = convNetDataset(
        test_attack, test_benign, transform=min_max_transform, sample_type=sys.argv[1]
    )

    # Create DataLoader
    sensitivity_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )

    # Setup model, loss function, and optimizer, load model
    model = convNet().to(device)
    state_dict = torch.load('/s/bach/b/class/cs535/cs535b/ddos-classification/1DCNN_model', map_location=torch.device("cpu"))
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
