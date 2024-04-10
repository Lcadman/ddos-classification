import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50

def setup(rank, world_size):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    # Properly shutdown process
    dist.destroy_process_group()

def prepare_dataset(batch_size):
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset and distributed samplers
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    return train_loader

def train(rank, world_size):
    # Run setup and initalize distributed model and training loader
    setup(rank, world_size)
    model = resnet50().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    train_loader = prepare_dataset(batch_size=32)

    # Setup optimizer and loss functions
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss().to(rank)

    # Loop over the number of epochs
    for epoch in range(10):
        # Put model in training mode, loop over samples in train loader
        ddp_model.train()
        train_loader.sampler.set_epoch(epoch)
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # Perform distributed training logic
            inputs = inputs.to(rank)
            labels = labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print the loss
            if i % 10 == 0 and rank == 0:
                print(f"[{epoch}, {i}] loss: {loss.item()}")

    # Run cleanup before exiting
    cleanup()

def main():
    # Get the environment variables from slurm 
    rank = int(os.getenv("SLURM_PROCID", "0"))
    world_size = int(os.getenv("SLURM_NTASKS", "1"))
    local_rank = int(os.getenv("SLURM_LOCALID", "0"))

    # Set the device and run train
    torch.cuda.set_device(local_rank)
    train(rank, world_size)

# Run main function
if __name__=="__main__":
    main()
