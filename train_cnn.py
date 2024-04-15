import torch
import torch.optim as optim
from tqdm import tqdm
from cnn import HandGestureCNN
from dataset import HandPoseDataset, get_dataloader
from torch.utils.data import random_split
from matplotlib import pyplot as plt
import numpy as np
from constants import *

# # Create the dataloader
dataloader = get_dataloader("/home/smeds/Desktop/VisGAN/hand_poses_dataset_CROP", batch_size)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # Define the device for training

if device.type == 'cuda':
    print(f'Device selected: {torch.cuda.get_device_name(0)}')
    torch.cuda.manual_seed_all(seed)
else:
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        torch.mps.manual_seed_all(seed)
        print(x)
    else:
        print("MPS device not found.")

# Seed 
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize the CNN model
netCNN = HandGestureCNN(nc, ndf, num_classes).to(device)

# Initialize CrossEntropyLoss function
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.NLLLoss()

# Setup Adam optimizer for the CNN
optimizerCNN = optim.Adam(netCNN.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # Create a progress bar
    progress_bar = tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    # For each batch in the dataloader
    for i, data in progress_bar:
        # Move the input data to the device
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizerCNN.zero_grad()

        # Forward pass
        outputs = netCNN(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizerCNN.step()

        # Print statistics
        losses.append(loss.item())
        if i == 100 :
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch, num_epochs, i, len(dataloader), loss.item()))
            i = 0
        else:
            i += 1

        # Update the progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        iters += 1

    print("Epoch Finished")
    print(f"Loss: {loss.item():.4f}")

# Save the CNN model
print("Saving the CNN model")
torch.save(netCNN.state_dict(), 'cnn_model.pth')

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the loss
ax.plot(losses)

# Set the labels
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")

# Show the plot
plt.show()
