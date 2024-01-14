# Feel free to change / extend / adapt this source code as needed to complete the homework, based on its requirements.
# This code is given as a starting point.
#
# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations, 
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
batch_size = 8
max_num_epoch = 300
hps = {'lr':0.01}

# ---- options ----
DEVICE_ID = 'cpu' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False

# --- imports ---
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils
from utils import read_image
import sys
from skimage import io


torch.multiprocessing.set_start_method('spawn', force=True)

# ---- utility functions -----
def get_loaders(batch_size,device):
    data_root = 'ceng483-hw3-dataset' 
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader

class AccuracyLoss(nn.Module):
    def __init__(self, error_margin=12):
        super(AccuracyLoss, self).__init__()
        self.error_margin = error_margin

    def forward(self, ground_truth, prediction):
        # Ensure the images have the same shape
        assert ground_truth.shape == prediction.shape, "Ground truth and prediction shapes must match"
        
        # Convert to PyTorch tensors
        ground_truth = 255*(ground_truth+1)/2
        ground_truth = ground_truth.view(-1).long()
        prediction = 255*(prediction+1)/2
        prediction = prediction.view(-1).long()
        
        # Compute the accuracy
        cur_acc = torch.sum(torch.abs(ground_truth - prediction) < self.error_margin).float() / ground_truth.shape[0]

        # Compute the ratio of correctly estimated pixels to all pixels
        return 1.0 - cur_acc  # Returning 1.0 - accuracy as a loss

# ---- ConvNet -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Decoder
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # Adding ReLU activation after BatchNorm
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
            )
            

    def forward(self, x):
        x = self.model(x)
        return x
    

# ---- training code -----
device = torch.device(DEVICE_ID)
print('device: ' + str(device))
net = Net().to(device=device)
criterion = nn.MSELoss()
criteriontwo = AccuracyLoss()
optimizer = optim.SGD(net.parameters(), lr=hps['lr'])
train_loader, val_loader = get_loaders(batch_size,device)

min_valid_loss = np.inf

'''if LOAD_CHKPT:
    print('loading the model from the checkpoint')
    model.load_state_dict(os.path.join(LOG_DIR,'checkpoint.pt'))'''

print('training begins')

hist_train = []
hist_val = []

hist_acc = []

early_stopping_patience = 10
best_val_loss = float('inf')
current_patience = 0
improvement_threshold=0.1


for epoch in range(max_num_epoch):  
    running_loss = 0.0 # training loss of the network
    valid_loss = 0.0
    running_acc = 0.0
    for iteri, data in enumerate(train_loader, 0):
        inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.

        optimizer.zero_grad() # zero the parameter gradients

        # do forward, backward, SGD step
        preds = net(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        

        '''# print loss
        
        print_n = 100 # feel free to change this constant
        if iteri % print_n == (print_n-1):    # print every print_n mini-batches
            print('[%d, %5d] network-loss: %.3f' %
                  (epoch + 1, iteri + 1, running_loss / 100))
            running_loss = 0.0
            # note: you most probably want to track the progress on the validation set as well (needs to be implemented)'''

        if (iteri==0) and VISUALIZE: 
            hw3utils.visualize_batch(inputs,preds,targets)

    ### CHANGE ----------------
            
    for iteri,data in enumerate(val_loader,0):
        inputs,targets = data

        prediction = net(inputs)

        vloss = criterion(prediction,targets)
        vacc = criteriontwo(prediction,targets)
        valid_loss += vloss.item() * inputs.size(0)
        running_acc += vacc.item() * inputs.size(0)
        
    ### CHANGE ----------------
        
    hist_train.append(running_loss/len(train_loader))
    hist_val.append(valid_loss / len(val_loader))
    hist_acc.append(running_acc/len(val_loader))

    val_loss = valid_loss / len(val_loader)

    if val_loss < (1 - improvement_threshold) * best_val_loss:
        best_val_loss = val_loss
        current_patience = 0
    else:
        current_patience += 1
        if current_patience >= early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
        
    print(f'Epoch {epoch+1} \t\t Training Loss: {running_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)} \t\t Accuracy Loss: {running_acc/len(val_loader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss


    print('Saving the model, end of epoch %d' % (epoch+1))
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    torch.save(net.state_dict(), os.path.join(LOG_DIR,'checkpoint5.pt'))
    #hw3utils.visualize_batch(inputs,preds,targets,os.path.join(LOG_DIR,'example.png'))

print('Finished Training')


def preprocess_image(image_path):
    # Read the image
    image = io.imread(image_path)

    # Convert to torch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust normalization values if needed
    ])
    input_tensor = transform(image)

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor

folder_path = "images_grayscale"

# Get a list of all files in the folder
all_files = os.listdir(folder_path)

# Filter out only image files (you may need to customize the image file extensions)
image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
estimations = []
for image_path in image_files:
    # Preprocess the image
    input_tensor = preprocess_image("images_grayscale/"+image_path)

    # Make inference
    with torch.no_grad():
        output = net(input_tensor)
        estimations.append(output.numpy())


# Write image names to img_names.txt
with open('test_images.txt', 'w') as file:
    for image_name in image_files:
        file.write(image_name + '\n')

# Convert the list of predictions to a numpy array
predictions_np = np.concatenate(estimations, axis=0)
predictions_np = ((predictions_np + 1) / 2 * 255).astype(np.uint8)
predictions_np = np.moveaxis(predictions_np, 1, -1)

np.save("estimations_test.npy", predictions_np)
