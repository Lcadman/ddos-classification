import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from SetupInfo import SlurmSetup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.data.distributed import DistributedSampler as ds

# Define constants
BATCH_SIZE = 1000
EPOCH_COUNT = 5
LEARNING_RATE = 0.001

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(self, attack_file, benign_file, transform=None):
        # Load data from files
        self.attack_data = pd.read_csv(attack_file)
        self.benign_data = pd.read_csv(benign_file)

        # # For binary classification, modify the labels to be integers
        self.benign_data.iloc[:, -1] = 0
        self.attack_data.iloc[:, -1] = 1

        # Oversample the BENIGN data by a factor of ten
        self.benign_data = pd.concat([self.benign_data] * 10, ignore_index=True)

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
        if not hasattr(self, "fc1"):
            # Initialize the fc1 layer dynamically
            self.fc1 = nn.Linear(num_features, 100).to(x.device)
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def train(model, train_loader, criterion, optimizer):
    # Set the model to training mode, save variables for tracking metrics
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop over train loader
    for data, labels in train_loader:
        # Ensure labels are in the correct shape
        data, labels = data.float().to(device), labels.float().unsqueeze(1).to(device)

        # Reset gradients each batch
        optimizer.zero_grad()

        # Forward pass on batch
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Ensure outputs and labels are floats
        outputs = outputs.type(torch.float32)
        labels = labels.type(torch.float32)

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Invalid loss detected")

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
        for data, labels in test_loader:
            # Ensure labels are in the correct shape
            data, labels = data.float().to(device), labels.float().unsqueeze(1).to(
                device
            )

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
    # Setup Slurm
    setup = SlurmSetup()
    print(f"rank {setup.rank}: starting communication")
    setup.establish_communication()
    print(f"rank {setup.rank}: communication established")

    # Set device
    print(f"rank {setup.rank}: device is {device}")

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
    train_dataset = convNetDataset(
        train_attack, train_benign, transform=min_max_transform
    )
    test_dataset = convNetDataset(test_attack, test_benign, transform=min_max_transform)

    # Create sampler for distributed training
    train_sampler = ds(train_dataset, num_replicas=setup.world_size, rank=setup.rank)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Setup model, loss function, and optimizer
    model = convNet().to(device)
    model = ddp(model).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loop over each epoch
    for epoch in range(EPOCH_COUNT):

        # Set the epoch for the sampler
        train_sampler.set_epoch(epoch)

        # Train the model and print loss and accuracy
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        print(
            f"rank {setup.rank}: Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )

        # Test the model on main node
        if setup.is_main_process():
            # Test the model and print loss and accuracy
            test_loss, test_accuracy = test(model, test_loader, criterion)
            print(
                f"rank {setup.rank}: Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
            )

    # Save the model
    if setup.is_main_process():
        torch.save(model.state_dict(), f"1DCNN_model")


# Run main function
if __name__ == "__main__":
    print("Starting training...")
    main()
