from data_provider import DataContainer
from data_augmentation import Functions
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pylab import  plot, show, axis, arange, figure, uint8


data_container = DataContainer()
image_array, _ = data_container.training.get_range(0, 1)
brg_image = image_array[0]
rgb_image = cv2.cvtColor(brg_image, cv2.COLOR_BGR2RGB)
# Image data
image = rgb_image
#image = cv2.imread('imgur.png',0) # load as 1-channel 8bit grayscale

f = Functions()

r = f.add_noise(0)[0](image)

cv2.imshow('original image',image)
cv2.imshow("modified image", r)


closeWindow = -1
while closeWindow<0:
    closeWindow = cv2.waitKey(1)
cv2.destroyAllWindows()



#data_container = DataContainer()
#image_array, _ = data_container.training.get_range(0, 1)
#image = image_array[0]
#rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plt.imshow(rgb_image)
#plt.show()

#a = cv2.imread(r"E:\personal\std\_nd_projects\behavioral-cloning\captured_data\mask_black.png")
#s = cv2.addWeighted(rgb_image, 0.7, a, 1.0, 0)
#plt.imshow(s)
#plt.show()
