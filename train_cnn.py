import torch
import torch.optim as optim
from tqdm import tqdm
from cnn import HandGestureCNN
from dataset import HandPoseDataset, get_dataloader
from torch.utils.data import random_split

# Hyperparameters
batch_size = 32
nc = 3  # Number of channels in the training images. For color images this is 3
ndf = 64  # Size of feature maps in the discriminator
num_epochs = 5
# num_epochs = 10
lr = 0.0002
beta1 = 0.5
# beta1 = 0.7
num_classes = 20

# # Create the dataloader
dataloader = get_dataloader("train_dataset", batch_size)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # Define the device for training

if device.type == 'cuda':
    print(f'Device selected: {torch.cuda.get_device_name(0)}')
else:
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

# Initialize the CNN model
netCNN = HandGestureCNN(nc, ndf, num_classes).to(device)

# Initialize CrossEntropyLoss function
criterion = torch.nn.CrossEntropyLoss()

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
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch, num_epochs, i, len(dataloader), loss.item()))

        # Update the progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        iters += 1

    print("Epoch Finished")
    print(f"Loss: {loss.item():.4f}")

# Save the CNN model
print("Saving the CNN model")
torch.save(netCNN.state_dict(), 'cnn_model_1.pth')
