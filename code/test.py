from data_provider import DataContainer
from data_augmentation import flip_center_image
from data_augmentation import histograms_equalization
import matplotlib.pyplot as plt
import cv2
import numpy as np


data_container = DataContainer()
image_array, _ = data_container.training.get_range(0, 1)
image = image_array[0]
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plt.imshow(rgb_image)
#plt.show()

a = cv2.imread(r"E:\personal\std\_nd_projects\behavioral-cloning\captured_data\mask_black.png")
s = cv2.addWeighted(rgb_image, 0.7, a, 1.0, 0)
plt.imshow(s)
plt.show()
