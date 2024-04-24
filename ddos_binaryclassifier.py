import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from tqdm import tqdm


device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
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


class BinaryClassificationDataset(Dataset):
    def __init__(self, attack_file, benign_file, transform=None):
        # Load data from files
        self.attack_data = pd.read_csv(attack_file)
        self.benign_data = pd.read_csv(benign_file)

        # Combine the datasets
        self.data = pd.concat([self.attack_data, self.benign_data], ignore_index=True)

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
        label = self.data.iloc[idx, -1].astype("int")

        # Transform the samples if present
        if self.transform:
            sample = self.transform(sample)

        # Return the samples and labels as torch tensors
        # TODO maybe this if data is not rightly typed: return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)
        return torch.tensor(sample), torch.tensor(label)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(79, 128)  # Input layer to first hidden layer
        self.relu = nn.ReLU()             # Activation function
        self.layer2 = nn.Linear(128, 64)  # Second hidden layer
        self.output_layer = nn.Linear(64, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()       # Sigmoid activation for binary output

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x


def train(model, train_loader, criterion, optimizer):
    # Set the model to training mode, save variables for tracking metrics
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop over train loader
    for data, labels in tqdm(train_loader):
        # Ensure labels are in the correct shape
        data, labels = data.float().to(device), labels.float().unsqueeze(1).to(device)

        # Reset gradients each batch
        optimizer.zero_grad()

        # Forward pass on batch
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass and step optimizer
        loss.backward()
        optimizer.step()

        # Calculate loss and accuracy, store correct predictions
        running_loss += loss.item()
        predicted = outputs.round()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Return the average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def test(model, test_loader, criterion):
    # Set the model to evaluation mode, save variables for tracking metrics
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient updates, loop over test loader
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
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

    # Return the average loss and accuracy
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


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

    # Create dataset instances
    train_dataset = BinaryClassificationDataset(
        train_attack, train_benign, transform=min_max_transform
    )
    test_dataset = BinaryClassificationDataset(
        test_attack, test_benign, transform=min_max_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=1000, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1000, shuffle=False, num_workers=4
    )

    # Setup model, loss function, and optimizer
    model = BinaryClassifier()
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loop over each epoch
    for epoch in range(5):
        # Train the model and print loss and accuracy
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )

        # Test the model and print loss and accuracy
        test_loss, test_accuracy = test(model, test_loader, criterion)
        print(
            f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
        )


# Run main function
if __name__ == "__main__":
    print("Starting training...")
    main()
