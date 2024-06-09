import numpy as np
import cv2 
import matplotlib.pyplot as plt

path=r"/home/summy/Tesis/preprocessing_images/buenos_FPS/"
complete_path=path + '0095.jpg'
image = cv2.imread(complete_path, cv2.IMREAD_COLOR)
image_copy = image.copy()

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



lower_bound = 30 #20#30 
upper_bound = 220 #250

"""

# Create a mask based on the grayscale intensity range
mask = cv2.inRange(gray_image, lower_bound, upper_bound)
roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
image_mask=gray_image*mask

equ = cv2.equalizeHist(image_mask)

new_image=cv2.cvtColor(equ,cv2.COLOR_GRAY2RGB)

new_image1=cv2.cvtColor(gray_image,cv2.COLOR_GRAY2RGB)

cv2.imshow('equ',equ)
cv2.imshow('mask',image_mask)
cv2.imshow('roi',roi)


cv2.waitKey(0)
cv2.destroyAllWindows()

"""
condition = (image[:,:,0] > 200) & (image[:,:,1] > 200) & (image[:,:,2] > 200)
image_copy[condition] = [255, 255, 255]
kernel = np.ones((3, 3), np.float32) / 9
image_copy = cv2.filter2D(image_copy, -1, kernel, borderType=cv2.BORDER_REPLICATE)
yuv_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2YUV)
result_image=yuv_image.copy()
result_image1=yuv_image.copy()



# lower_bound = 50
# upper_bound = 250

# # Create a mask based on the grayscale intensity range
mask = cv2.inRange(yuv_image[:, :, 0], lower_bound, upper_bound)
roi = cv2.bitwise_and(mask, gray_image, mask=mask)
#image_mask=yuv_image[:, :, 0]*mask

# # Apply histogram equalization to the Y channel
# image_mask[:, :, 0] = cv2.equalizeHist(image_mask[:, :, 0])
equalized_roi= cv2.equalizeHist(roi)

result_image[:,:,0] = np.where(mask == 0, yuv_image[:, :, 0], equalized_roi)
result_image1[:,:,0] = cv2.bilateralFilter(result_image[:,:,0],3,75,75)


recovered_image = cv2.cvtColor(result_image, cv2.COLOR_YUV2BGR)
recovered_image1 = cv2.cvtColor(result_image1, cv2.COLOR_YUV2BGR)


# # Convert the image back to RGB color space
# equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

# Display the original and equalized images
cv2.imshow('Original Image', image)
cv2.imshow('corrected colors', image_copy)
cv2.imshow('mask', mask)


cv2.imshow('y channel', yuv_image[:,:,0])
cv2.imshow('equalized roi',equalized_roi)
cv2.imshow('recovered_image',recovered_image)
cv2.imshow('recovered_image1',recovered_image1)

cv2.imwrite('corrected_image.jpg', recovered_image1)

cv2.waitKey(0)
cv2.destroyAllWindows()

