from PIL import Image
import os
import cv2
import numpy as np
# img = cv2.resize(img, (256, 256))

# def split_image_and_mask(image_path, mask_path, output_dir, tile_size, overlap):
#     # Read the image and mask using cv2
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (256, 256))
#     mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#     height, width = img.shape[:2]

#     # Ensure the mask has the same size as the image
#     if mask.shape[:2] != (height, width):
#         raise ValueError("The size of the image and the mask must be the same.")

#     # Calculate the step size based on the overlap
#     step_x = tile_size[0] - overlap
#     step_y = tile_size[1] - overlap

#     # Create output directories if they don't exist
#     img_output_dir = os.path.join(output_dir, 'images')
#     mask_output_dir = os.path.join(output_dir, 'masks')
#     os.makedirs(img_output_dir, exist_ok=True)
#     os.makedirs(mask_output_dir, exist_ok=True)

#     tile_num = 0
#     y = 0
#     while y < height:
#         x = 0
#         while x < width:
#             # Calculate the region to crop
#             left = x
#             upper = y
#             right = min(x + tile_size[0], width)
#             lower = min(y + tile_size[1], height)

#             # Adjust the starting points for the last tiles to ensure correct size
#             if right == width:
#                 left = max(width - tile_size[0], 0)
#             if lower == height:
#                 upper = max(height - tile_size[1], 0)

#             # Crop the image and mask
#             img_tile = img[upper:upper + tile_size[1], left:left + tile_size[0]]
#             mask_tile = mask[upper:upper + tile_size[1], left:left + tile_size[0]]

#             # Save the tiles
#             cv2.imwrite(os.path.join(img_output_dir, f"tile_{tile_num}.png"), img_tile)
#             cv2.imwrite(os.path.join(mask_output_dir, f"mask_{tile_num}.png"), mask_tile)
#             tile_num += 1

#             # Move to the next tile position
#             x += step_x

#         # Move to the next row of tiles
#         y += step_y

def visualize_image_and_mask(image_path, mask_path, color=[255, 0, 0], alpha=0.5):
    import matplotlib.pyplot as plt
    """
    Visualizes an image with its segmentation mask overlay.

    Parameters:
    - image_path (str): Path to the input image.
    - mask_path (str): Path to the segmentation mask.
    - color (list): RGB color for the mask overlay (default is red).
    - alpha (float): Transparency factor for the overlay (default is 0.5).

    Returns:
    - None
    """
    # Read the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Optionally resize the mask to match the image dimensions
    # mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Create an overlay by setting the mask color
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask > 0] = color

    # Combine the image and the overlay
    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Plot the original image, mask, and the output image with the overlay
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Segmentation Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Image with Mask Overlay')
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def split_image_and_mask(image_path, mask_path, output_dir, tile_size, overlap):
    # Read the image and mask using cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]

    # Ensure the mask has the same size as the image
    if mask.shape[:2] != (height, width):
        raise ValueError("The size of the image and the mask must be the same.")

    # Calculate the step size based on the overlap
    step_x = tile_size[0] - overlap
    step_y = tile_size[1] - overlap

    # Create output directories if they don't exist
    img_output_dir = os.path.join(output_dir, 'images')
    mask_output_dir = os.path.join(output_dir, 'masks')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # List to save the tile coordinates and dimensions
    tile_info = []

    tile_num = 0
    y = 0
    while y < height:
        x = 0
        while x < width:
            # Calculate the region to crop
            left = x
            upper = y
            right = min(x + tile_size[0], width)
            lower = min(y + tile_size[1], height)

            # Adjust the starting points for the last tiles to ensure correct size
            if right == width:
                left = max(width - tile_size[0], 0)
            if lower == height:
                upper = max(height - tile_size[1], 0)

            # Crop the image and mask
            img_tile = img[upper:upper + tile_size[1], left:left + tile_size[0]]
            mask_tile = mask[upper:upper + tile_size[1], left:left + tile_size[0]]

            # Save the tiles
            base_name = os.path.basename(image_path).split('.')[0]
            img_tile_filename = os.path.join(img_output_dir, f"{base_name}_tile_{tile_num}.png")
            mask_tile_filename = os.path.join(mask_output_dir, f"{base_name}_tile_{tile_num}.png")
            cv2.imwrite(img_tile_filename, img_tile)
            cv2.imwrite(mask_tile_filename, mask_tile)

            # Store the tile information
            tile_info.append({
                'tile_num': tile_num,
                'img_tile_filename': img_tile_filename,
                'mask_tile_filename': mask_tile_filename,
                'left': left,
                'upper': upper,
                'right': right,
                'lower': lower
            })

            tile_num += 1

            # Move to the next tile position
            x += step_x

        # Move to the next row of tiles
        y += step_y

    return tile_info, (width, height)

def merge_tiles(tile_info, output_img_path, output_mask_path, original_size):
    width, height = original_size

    # Create blank images for the output
    merged_img = np.zeros((height, width, 3), dtype=np.uint8)
    merged_mask = np.zeros((height, width), dtype=np.uint8)

    for tile in tile_info:
        # Load the tile images
        img_tile = cv2.imread(tile['img_tile_filename'])
        mask_tile = cv2.imread(tile['mask_tile_filename'], cv2.IMREAD_UNCHANGED)
        
        # If mask_tile has 3 channels, convert it to grayscale
        if len(mask_tile.shape) == 3 and mask_tile.shape[2] == 3:
            mask_tile = cv2.cvtColor(mask_tile, cv2.COLOR_BGR2GRAY)

        # Get the position to place this tile
        left, upper = tile['left'], tile['upper']

        # Place the tile in the correct position
        merged_img[upper:upper + img_tile.shape[0], left:left + img_tile.shape[1]] = img_tile
        merged_mask[upper:upper + mask_tile.shape[0], left:left + mask_tile.shape[1]] = mask_tile

    # Save the merged image and mask
    cv2.imwrite(output_img_path, merged_img)
    cv2.imwrite(output_mask_path, merged_mask)


# tile_info, original_size = split_image_and_mask('./COVID-19_Radiography_Dataset/COVID/images/COVID-2.png', \
#                      './COVID-19_Radiography_Dataset/COVID/masks/COVID-2.png', 'output_tiles', (64, 64), 32)


# merged_mask_path = './mask_test.png'
# merged_image_path = './img_test.png'
# merge_tiles(tile_info, merged_image_path, merged_mask_path, original_size)

# visualize_image_and_mask('./COVID-19_Radiography_Dataset/COVID/images/COVID-2.png', \
#                          './COVID-19_Radiography_Dataset/COVID/masks/COVID-2.png')

from tqdm import tqdm
from glob import glob

path = '/home/chaunm/Projects/dataset/COVID-19_Radiography_Dataset/COVID/'
saved_path = '/home/chaunm/Projects/dataset/seg_tiling/COVID/'
img_list = glob(os.path.join(path, 'images', '*.png'))
mask_list = glob(os.path.join(path, 'masks', '*.png'))


for (img_path, mask_path) in tqdm(zip(img_list, mask_list)):
    tile_info, original_size = split_image_and_mask(img_path, mask_path, saved_path, (64, 64), 16)
    