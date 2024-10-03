# Description: This script trains the CNN model on the dataset.

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from cnn import HandGestureCNN
from utils.dataset import get_dataloader
from matplotlib import pyplot as plt
from utils.constants import *

#create the dataloader
dataloader = get_dataloader(train_path, batch_size)

#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # Define the device for training

if device.type == 'cuda':
    print(f'Device selected: {torch.cuda.get_device_name(0)}')
    torch.cuda.manual_seed_all(seed)
else:
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        torch.mps.manual_seed(seed)
        print(x)
    else:
        print("MPS device not found.")

#seed 
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#CNN model
netCNN = HandGestureCNN(nc, ndf, num_classes).to(device)

#CrossEntropyLoss function
criterion = torch.nn.CrossEntropyLoss()

#Adam optimizer for the CNN
optimizerCNN = optim.Adam(netCNN.parameters(), lr=lr, betas=(beta1, 0.999))

######### Training Loop #########

#lists to keep track of progress
avg_losses_per_epoch = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    #progress bar
    progress_bar = tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    #for each batch in the dataloader
    epoch_losses = []  # Store losses for each batch in this epoch
    for i, data in progress_bar:
        inputs, labels = data[0].to(device), data[1].to(device)     #move the input data to the device

        optimizerCNN.zero_grad()    #zero the parameter gradients

        ##### forward pass #####
        outputs = netCNN(inputs)
        loss = criterion(outputs, labels)

        ##### backward and optimize #####
        loss.backward()
        optimizerCNN.step()

        #statistics prints
        epoch_losses.append(loss.item())
        if i == 100 :
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch, num_epochs, i, len(dataloader), loss.item()))
            i = 0

        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        iters += 1

    avg_loss_this_epoch = sum(epoch_losses) / len(epoch_losses)
    avg_losses_per_epoch.append(avg_loss_this_epoch)
    print("Epoch Finished")
    print(f"Average Loss: {avg_loss_this_epoch:.4f}")

#save the CNN model
print("Saving the CNN model")
torch.save(netCNN.state_dict(), pth)

#plot the loss
fig, ax = plt.subplots()
ax.plot(range(1, num_epochs + 1), avg_losses_per_epoch)
ax.set_xlabel("Epoch")
ax.set_ylabel("Average Loss")
print("Saving the loss plot")
plt.savefig('/Users/vale/Desktop/Sapienza/Vision/images/loss_plot_2.png')
plt.show()
