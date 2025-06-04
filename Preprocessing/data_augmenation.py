import cv2
import os
import numpy as np


############

def corlorCorrection(frame,margin=40):

    """
    This function corrects the background of every frame, so it is 
    uniform and white
    
    """

    # lower_bound = 30 #20#30 
    # upper_bound = 220 #250

    image_copy = frame.copy() #creating a copy of the frame
    h, w, _ = frame.shape

    condition = (frame[:,:,0] > 200) & (frame[:,:,1] > 200) & (frame[:,:,2] > 200) #looking for sections where the pixel are closer to white
                                                                                   #selecting the greay sections
    image_copy[condition] = [255, 255, 255] # changes the color of every pixel that satisfies the condition

    # kernel = np.ones((3, 3), np.uint8)
    # image_copy = cv2.morphologyEx(image_copy, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((3, 3), np.float32) / 9  #creating a kernel of ones 3x3
    # image_copy = cv2.filter2D(image_copy, -1, kernel, borderType=cv2.BORDER_REPLICATE) #applyig the filter to the borders-->smooth transition
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[margin:h-margin, margin:w-margin] = 1  # Exclude the central area

    # Define the color range for near-white (light gray) colors
    lower_bound = np.array([180, 180, 180], dtype=np.uint8)
    upper_bound = np.array([255, 255, 255], dtype=np.uint8)

    # Create a mask for near-white areas within the border region only
    near_white_mask = cv2.inRange(frame, lower_bound, upper_bound)
    near_white_mask = cv2.bitwise_and(near_white_mask, near_white_mask, mask=(1 - mask))

    # Change near-white regions to pure white in the border area
    image_copy[near_white_mask > 0] = [255, 255, 255]


    return image_copy


def rotate_image(image, angle, background_color=(255, 255, 255)):
    """
    This function applies rotation to every frame of the video, given an angle value
    """
    # Get the image size
    h, w = image.shape[:2]
    
    # Compute the rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation with white background
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
    return rotated_image

def zoom_image(image, scale=1.2, background_color=(255, 255, 255)):

    """
    This function applies zoom variation to every frame of the video given an scale factor
    """
    # Compute the size for zooming in or out
    h, w = image.shape[:2]
    zoomed_size = (int(w * scale), int(h * scale))
    
    # Resize the image
    zoomed_image = cv2.resize(image, zoomed_size, interpolation=cv2.INTER_LINEAR)
    
    # Center-crop or pad the image back to original size
    if scale > 1.0:
        # Zoom-in: crop the center
        start_x = (zoomed_image.shape[1] - w) // 2
        start_y = (zoomed_image.shape[0] - h) // 2
        cropped_image = zoomed_image[start_y:start_y + h, start_x:start_x + w]
    else:
        # Zoom-out: add padding
        padded_image = np.full((h, w, 3), background_color, dtype=np.uint8)
        offset_x = (w - zoomed_image.shape[1]) // 2
        offset_y = (h - zoomed_image.shape[0]) // 2
        padded_image[offset_y:offset_y + zoomed_image.shape[0], offset_x:offset_x + zoomed_image.shape[1]] = zoomed_image
        cropped_image = padded_image

    return cropped_image

def translate_image(image, x_shift=0, y_shift=0, background_color=(255, 255, 255)):

    """
    This function applies translation to every frame of the video given an x and y shift
    """
    # Get the dimensions of the image
    h, w = image.shape[:2]
    
    # Define the translation matrix
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    
    # Apply the translation
    translated_image = cv2.warpAffine(image, translation_matrix, (w, h), 
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
    return translated_image

def add_salt_and_pepper_noise(image, noise_ratio=0.02):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)
 
    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = [0, 0, 0] 
        else:
            noisy_image[row, col] = [255, 255, 255]
 
    return noisy_image

def shear_image(image, x_shear=0, y_shear=0, background_color=(255, 255, 255)):
    # Get the original dimensions of the image
    h, w = image.shape[:2]

    # Define the shear matrix
    shear_matrix = np.float32([[1, x_shear, 0], [y_shear, 1, 0]])

    # Calculate the bounding box dimensions to apply shear without resizing
    # Warp the image with a slightly larger canvas to avoid black edges
    temp_w = int(w + abs(y_shear) * h)
    temp_h = int(h + abs(x_shear) * w)

    # Apply the shear transformation with a larger canvas
    sheared_image = cv2.warpAffine(image, shear_matrix, (temp_w, temp_h), 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
    
    # Center-crop the image back to original dimensions
    start_x = (sheared_image.shape[1] - w) // 2
    start_y = (sheared_image.shape[0] - h) // 2
    cropped_sheared_image = sheared_image[start_y:start_y + h, start_x:start_x + w]
    
    return cropped_sheared_image
