### https://www.geeksforgeeks.org/create-a-vignette-filter-using-python-opencv/

import numpy as np 
import cv2 

#reading the image
input_image = cv2.imread('test2_filtro.jpg') 

#resizing the image according to our need
# resize() function takes 2 parameters,
# the image and the dimensions
#input_image = cv2.resize(input_image), (480, 480)) 

# ========================================================================
# Extracting the height and width of an image
rows, cols = input_image.shape[:2] 

# generating vignette mask using Gaussian
# resultant_kernels 
X_resultant_kernel = cv2.getGaussianKernel(cols,200) 
Y_resultant_kernel = cv2.getGaussianKernel(rows,200) 

#generating resultant_kernel matrix
resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T 

#creating mask and normalising by using np.linalg 
# function 
mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel) 
vignette = np.copy(input_image) 

# applying the mask to each channel in the input image 
for i in range(3): 
    vignette[:,:,i] = vignette[:,:,i] * mask 

#displaying the orignal image
cv2.imshow('Original', input_image) 

#displaying the vignette filter image
# yo cv2.imwrite('food_vignette.png', vignette)
cv2.imshow('VIGNETTE', vignette) 

#========================================================================
# Extracting the height and width of an image
rows, cols = vignette.shape[:2] 

# generating vignette mask using Gaussian
# resultant_kernels 
X_resultant_kernel = cv2.getGaussianKernel(cols,200) 
Y_resultant_kernel = cv2.getGaussianKernel(rows,200) 
 
mask = np.linalg.norm(resultant_kernel) / (255 * resultant_kernel)
output = np.copy(vignette) 

for i in range(3): 
    output[:,:,i] = output[:,:,i] * mask 

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    cv2.imshow('FINAL', output) 
# yo cv2.imwrite('food_final.png', output)

# Maintain output window utill
# user presses a key
cv2.waitKey(0) 

# Destroying present windows on screen
cv2.destroyAllWindows()
