import numpy as np
import cv2 
import matplotlib.pyplot as plt

path=r"/home/summy/Tesis/preprocessing_images/buenos_FPS/"
complete_path=path + '0095.jpg'

# complete_path="/home/summy/Tesis/preprocessing_images/corrected_image.jpg"
image_original = cv2.imread(complete_path, cv2.IMREAD_COLOR)
image_original=cv2.cvtColor(image_original,cv2.COLOR_BGR2RGB)

# image = cv2.imread(complete_path,0)

image=image_original.copy()
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_blur_g=cv2.GaussianBlur(image_gray,(5,5),0)
image_blur_b=cv2.bilateralFilter(image_gray,5,200,200)

image_thres=cv2.adaptiveThreshold(
    image_gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=5,
    C=2
)
image_thres_b=cv2.adaptiveThreshold(
    image_blur_b,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=5,
    C=2
)
image_thres_g=cv2.adaptiveThreshold(
    image_blur_g,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=5,
    C=2
)
# image_thres=cv2.bilateralFilter(image_thres,3,75,75)



image_thres=cv2.medianBlur(image_thres, ksize = 3)
image_thres_b=cv2.medianBlur(image_thres_b, ksize = 3)
image_thres_g=cv2.medianBlur(image_thres_g, ksize = 3)
image_thres_3c=cv2.cvtColor(image_thres,cv2.COLOR_GRAY2RGB)
image_thres_3c_b=cv2.cvtColor(image_thres_b,cv2.COLOR_GRAY2RGB)
image_thres_3c_g=cv2.cvtColor(image_thres_g,cv2.COLOR_GRAY2RGB)




image_canny_b=cv2.Canny(image_blur_g,50,70,apertureSize=3,L2gradient=True)
image_canny_g=cv2.Canny(image_blur_b,50,70,apertureSize=3,L2gradient=True)
image_canny_3c_b=255-cv2.cvtColor(image_canny_b,cv2.COLOR_GRAY2RGB)
image_canny_3c_g=255-cv2.cvtColor(image_canny_g,cv2.COLOR_GRAY2RGB)

# image_summed=cv2.add(image_canny_3c,image_original)
image_weighted=cv2.addWeighted(image_original,0.8,image_thres_3c,0.2,0)
image_weighted_b=cv2.addWeighted(image_original,0.8,image_thres_3c_b,0.2,0)
image_weighted_g=cv2.addWeighted(image_original,0.8,image_thres_3c_g,0.2,0)

image_weighted1_b=cv2.addWeighted(image_original,0.8,image_canny_3c_b,0.2,0)
image_weighted1_g=cv2.addWeighted(image_original,0.8,image_canny_3c_g,0.2,0)



prom_after=cv2.GaussianBlur(image_weighted,(3,3),0)

# new_image=image_original.copy()
# new_image[:,:,0] = cv2.Canny(new_image[:,:,0],50,80,apertureSize=3,L2gradient=True)
# new_image[:,:,1] = cv2.Canny(new_image[:,:,1],50,80,apertureSize=3,L2gradient=True)
# new_image[:,:,2] = cv2.Canny(new_image[:,:,2],50,80,apertureSize=3,L2gradient=True)



images=[image_original,image_weighted,image_weighted_b,image_weighted_g,image_weighted1_b,image_weighted1_g]
# images=[roberts_img,sobel_img,scharr_img,prewitt_img,farid_img]

names_images=['original','thres_nothing','thres_b','thres_g','canny_b','canny_g']
# names_images=['roberts_img','sobel_img','scharr_im','prewitt_img','farid_img']


for i in range(len(images)):
    plt.subplot(3,3, i+1)
    plt.imshow(images[i],cmap='gray',vmin=0,vmax=255)
    plt.title(names_images[i])
    plt.xticks([])
    plt.yticks([])
plt.show()


cv2.waitKey(0) 
cv2.destroyAllWindows() 