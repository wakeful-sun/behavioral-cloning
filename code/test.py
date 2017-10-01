#from data_provider import DataContainer
#from data_provider import DrivingDataSequence
#from data_augmentation import flip_center_image
#from data_augmentation import histograms_equalization
#import matplotlib.pyplot as plt


#data_container = DataContainer(0.1)
#image, _ = data_container.raw_data[1000].get_data()
#plt.imshow(image)
#plt.show()
#
#data_container.raw_data[1000].register_modification_func(histograms_equalization)
#image, _ = data_container.raw_data[1000].get_data()
#plt.imshow(image)
#plt.show()

class Test:

    def __init__(self, name):
        self.name = name
    
    def set_name(self, name):
        self.name = name

    @property
    def n(self):
        return self.name

a = [Test("a"), Test("b")]
b = a[:1]
b[0].set_name("x")
c = a + b

for i in c:
    print(i.n)