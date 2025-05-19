import cv2
import os
import albumentations as A
import numpy as np

# directory="/home/summy/Tesis/Temporal-segmentation-for-Peruvian-Sign-language-/preprocessing_images/buenos_FPS"

#############

# image = cv2.imread(os.path.join(directory,"0095.jpg"), cv2.IMREAD_COLOR)
# image = image.astype(np.float32) / 255.0

# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #just when using someting different from opencv


# single_transform = A.RandomBrightnessContrast(p=1.0)
# single_transform = A.Emboss(p=1.0)
# single_transform = A.PlanckianJitter(mode="blackbody", temperature_limit=(7999, 8050),sampling_method="uniform",p=1.0)
# single_transform = A.Compose([
#     A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1.0)
# ])

# single_transform = A.RandomGamma(gamma_limit=(150, 150), p=1.0) # adjust the brightness of an image while preserving the relative differences between darker and lighter areas
# single_transform=A.Sharpen(alpha=(0.5,0.5),p=1.0)
# single_transform=A.unsharp_mask(p=1.0)





    
# # Apply the transformation
# augmented = single_transform(image=image)
# augmented_image = augmented["image"]

# cv2.imshow("a",augmented_image)

# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

############

def corlorCorrection(frame,margin=40):

    # lower_bound = 30 #20#30 
    # upper_bound = 220 #250

    image_copy = frame.copy() #creating a copy of the frame
    h, w, _ = frame.shape

    ### Color Correction--> changing gray colors around the white background
    # lower_bound = np.array([180, 180, 180], dtype=np.uint8)  # Adjusted lower threshold for grayish white
    # upper_bound = np.array([255, 255, 255], dtype=np.uint8) 

    # mask = cv2.inRange(frame, lower_bound, upper_bound)

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

# def corlorCorrection(frame, margin=30):
#     # Create a copy of the frame
#     image_copy = frame.copy()
#     h, w, _ = frame.shape

#     # Define a region of interest (ROI) around the borders where gray lines may appear
#     mask = np.zeros((h, w), dtype=np.uint8)
#     mask[margin:h-margin, margin:w-margin] = 1  # Exclude the central area

#     # Define the color range for near-white (light gray) colors
#     lower_bound = np.array([180, 180, 180], dtype=np.uint8)
#     upper_bound = np.array([255, 255, 255], dtype=np.uint8)

#     # Create a mask for near-white areas within the border region only
#     near_white_mask = cv2.inRange(frame, lower_bound, upper_bound)
#     near_white_mask = cv2.bitwise_and(near_white_mask, near_white_mask, mask=(1 - mask))

#     # Change near-white regions to pure white in the border area
#     image_copy[near_white_mask > 0] = [255, 255, 255]

#     # Optionally, apply a slight Gaussian blur to only the modified areas for smoother transitions
#     blurred_image = cv2.GaussianBlur(image_copy, (3, 3), 0)
#     image_copy = np.where(near_white_mask[..., None] > 0, blurred_image, image_copy)

#     return image_copy

def rotate_image(image, angle, background_color=(255, 255, 255)):
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
# Load an image for testing


# image = cv2.imread(os.path.join(directory,"0099.jpg"), cv2.IMREAD_COLOR)


# # Rotate and zoom the image

# image=corlorCorrection(image)
# rotated_image = rotate_image(image, angle=0)
# zoomed_image = zoom_image(image, scale=1)
# # Translate the image 50 pixels to the right and 30 pixels down
# translated_right = translate_image(image, x_shift=15, y_shift=15)
# translated_left = translate_image(image, x_shift=-15, y_shift=-15)
# noisy_image=add_salt_and_pepper_noise(image, noise_ratio=0.001)

# # Apply shearing to the image
# sheared_image_x = shear_image(image, x_shear=0.1)  # Shear along the x-axis
# sheared_image_y = shear_image(image, y_shear=0.1)  # Shear along the y-axis

# # Display the results
# cv2.imshow("Sheared Image X-axis", sheared_image_x)
# cv2.imshow("Sheared Image Y-axis", sheared_image_y)

# # cv2.imshow("Noisy image", noisy_image)


# cv2.imshow("Translated Right", translated_right)
# cv2.imshow("Translated Left", translated_left)


# cv2.imshow("Rotated Image", rotated_image)
# cv2.imshow("Zoomed Image", zoomed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()