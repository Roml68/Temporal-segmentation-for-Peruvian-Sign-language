import numpy as np
import cv2 
import matplotlib.pyplot as plt

path=r"/home/summy/Tesis/preprocessing_images/buenos_FPS/"
complete_path=path + '0095.jpg'
image = cv2.imread(complete_path, cv2.IMREAD_COLOR)
image_copy = image.copy()

def corlorCorrection_and_histequalization(frame,lower_bound=40,upper_bound=220,color_domain="hsv", apply_filtering=False):

    # lower_bound = 30 #20#30 
    # upper_bound = 220 #250

    image_copy = frame.copy() #creating a copy of the frame

    ### Color Correction--> changing gray colors around the white background

    condition = (frame[:,:,0] > 200) & (frame[:,:,1] > 200) & (frame[:,:,2] > 200) #looking for sections where the pixel are closer to white
                                                                                   #selecting the greay sections
    image_copy[condition] = [255, 255, 255] # changes the color of every pixel that satisfies the condition
    kernel = np.ones((3, 3), np.float32) / 9  #creating a kernel of ones 3x3
    image_copy = cv2.filter2D(image_copy, -1, kernel, borderType=cv2.BORDER_REPLICATE) #applyig the filter to the borders-->smooth transition

    ### Changing color domain

    if color_domain=='yuv':

        new_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2YUV)

        result_image=new_image.copy() #making a copy
        mask_yuv= cv2.inRange(new_image[:, :, 0], lower_bound, upper_bound) # getting the section to apply hist equalization
        roi_yuv = cv2.bitwise_and(mask_yuv, new_image[:, :, 0], mask=mask_yuv) # creating the masl with 0's and 1's
        equalized_roi_yuv= cv2.equalizeHist(roi_yuv) #equalization
        result_image[:,:,0] = np.where(mask_yuv == 0, new_image[:, :, 0], equalized_roi_yuv)# reconstructing the image-->replacing

    elif color_domain=='hsv':

        new_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
        result_image=new_image.copy()
        mask_hsv = cv2.inRange(new_image[:, :, 2], lower_bound, upper_bound)
        roi_hsv = cv2.bitwise_and(mask_hsv, new_image[:, :, 2], mask=mask_hsv)
        equalized_roi_hsv= cv2.equalizeHist(roi_hsv)
        result_image[:,:,2] = np.where(mask_hsv== 0, new_image[:, :, 2], equalized_roi_hsv)

    elif color_domain=='compare':

        yuv_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2YUV)
        hsv_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)

        result_image_yuv=yuv_image.copy()
        result_image_hsv=hsv_image.copy()

        mask_yuv= cv2.inRange(yuv_image[:, :, 0], lower_bound, upper_bound)
        roi_yuv = cv2.bitwise_and(mask_yuv, yuv_image[:, :, 0], mask=mask_yuv)

        mask_hsv = cv2.inRange(hsv_image[:, :, 2], lower_bound, upper_bound)
        roi_hsv = cv2.bitwise_and(mask_hsv, hsv_image[:, :, 2], mask=mask_hsv)

        equalized_roi_yuv= cv2.equalizeHist(roi_yuv)
        equalized_roi_hsv= cv2.equalizeHist(roi_hsv)

        result_image_yuv[:,:,0] = np.where(mask_yuv== 0, yuv_image[:, :, 0], equalized_roi_yuv)
        result_image_hsv[:,:,2] = np.where(mask_hsv== 0, hsv_image[:, :, 2], equalized_roi_hsv)
    
    ### filtering to smooth and convert image back to BGR


    if apply_filtering:

        if color_domain=='yuv':

            result_image[:,:,0] = cv2.bilateralFilter(result_image[:,:,0],3,75,75)
            recovered_image = cv2.cvtColor(result_image, cv2.COLOR_YUV2BGR)
            
    
        elif color_domain=='hsv':

            result_image[:,:,2] = cv2.bilateralFilter(result_image[:,:,2],3,75,75)
            recovered_image = cv2.cvtColor(result_image, cv2.COLOR_HSV2BGR)

        elif color_domain=='compare':

            result_image_yuv[:,:,0] = cv2.bilateralFilter(result_image_yuv[:,:,0],3,75,75)
            result_image_hsv[:,:,2] = cv2.bilateralFilter(result_image_hsv[:,:,2],3,75,75)
            recovered_image_yuv = cv2.cvtColor(result_image_yuv, cv2.COLOR_YUV2BGR)
            recovered_image_hsv = cv2.cvtColor(result_image_hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('recovered_image_yuv',recovered_image_yuv)
            cv2.imshow('recovered_image_hsv',recovered_image_hsv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return recovered_image_hsv, recovered_image_yuv 
  
    else:

        if color_domain=='yuv':
            recovered_image = cv2.cvtColor(result_image, cv2.COLOR_YUV2BGR)

        elif color_domain=='hsv':
            recovered_image = cv2.cvtColor(result_image, cv2.COLOR_HSV2BGR)

        elif color_domain=='compare':
            recovered_image_yuv = cv2.cvtColor(result_image_yuv, cv2.COLOR_YUV2BGR)
            recovered_image_hsv = cv2.cvtColor(result_image_hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('recovered_image_yuv',recovered_image_yuv)
            cv2.imshow('recovered_image_hsv',recovered_image_hsv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return recovered_image_hsv, recovered_image_yuv 
        
    return recovered_image

# final_image_hsv,final_image_yuv=corlorCorrection_and_histequalization(image,lower_bound=40,upper_bound=220,color_domain="compare", apply_filtering=False)

# cv2.imwrite('equalized_image_hsv.jpg', final_image_hsv) 
# cv2.imwrite('equalized_image_yuv.jpg', final_image_yuv) 
