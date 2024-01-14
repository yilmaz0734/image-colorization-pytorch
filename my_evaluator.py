from utils import read_image
import sys
import numpy as np

# Folder path containing the images
folder_path = "images_grayscale"

# Read image names from images.txt
with open('images.txt', 'r') as file:
    image_names = [line.strip() for line in file]

image_array = [read_image(file.rstrip()).reshape(-1).astype(np.int64) for file in image_names]
image_array = np.concatenate(image_array,axis=0)
estimations = model(image_array)
estimations = ((estimations + 1) / 2 * 255).astype(np.uint8)
estimations = np.moveaxis(estimations, 1, -1)

print("Estimations shape: ",estimations.shape)


# Save the results to estimations.npy
np.save('estimations.npy', estimations)

print("Estimations saved to estimations.npy")