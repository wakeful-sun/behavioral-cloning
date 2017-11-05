## Behavioral Cloning

The project contains code for training neural network which can drive car in a simulator ([Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip), [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip), [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)).
Training data is not included. Project uses [**Keras**](https://keras.io/) library.

In order to collect images for training:
* run simulator in training mode
* press `RECORD` button and select output location
* press `RECORD` button again to start recording
* drive. Better to use wheel controller or gamepad. It will produce better quality training data.
* press `RECORD` button to stop recording
* put recorded images and CSV file into `.\captured_data\` folder

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

**Project structure and usage examples**
* `code` folder, contains all rquired code to train the network. `.\code\main.py` is the entry point. Here is an example of usage:
    
    ```
    python model.py --batch=25 --epochs=5 --dropout=0.5 --description=""
    ```

    All parameters are optional:
    * `batch` is a batch size (defauld value is 20)
    * `epochs` number of epochs (defaul value is 1)
    * `dropout` is a dropout rate for model's dropout layer (default value is 0.5)
    * `description` is run description (defaul value is empty text)
    
    Each `model.py` run will produce separate folder with `model.h5` file and detailed run information under `.\output\` location. The information in JSON format also will be added into `.\output\history.log` file. So it's easy to collect and compare different run results.
* `model.h5` is already trained Keras model that can drive car on the first track both forward and backward
* `forward.mp4` and `backward.mp4` files are video recordings of supplied `model.h5` neural network behavior
* `drive.py` file contains script that drives car in simulator in autonomous mode using pre-trained neural network. Here is an example of usage:
   * run simulator
   * select *AUTONOMOUS MODE*
   * execute `python drive.py model.h5` (`model.h5` is a path to pre-trained neural network)
   * enjoy :)

**Model Architecture and Training Strategy**

1. Model architecture

I utilazied LeNet model with extended dense layers. Here is it's configuration:

|Layer name              |  Layer type                 |Output Shape       |Param #    |
|========================|=============================|===================|===========|
|normalization           |Lambda                       |(None, 160, 320, 3)|0          |
|cropping                |Cropping2D                   |(None, 65, 320, 3) |0          |
|conv1_5x5_relu          |Conv2D 5x5 + RELU activation |(None, 61, 316, 6) |456        |
|max_pooling1_2x2        |MaxPooling 2x2               |(None, 30, 158, 6) |0          |
|conv2_5x5_relu          |Conv2D 5x5 + RELU activation |(None, 26, 154, 16)|2416       |
|max_pooling2_2x2        |MaxPooling 2x2               |(None, 13, 77, 16) |0          |
|conv3_5x5_relu          |Conv2D 5x5 + RELU activation |(None, 9, 73, 28)  |11228      |
|max_pooling3_2x2        |MaxPooling 2x2               |(None, 4, 36, 28)  |0          |
|flatten                 |Flatten                      |(None, 4032)       |0          |
|fully_connected_1       |Dense                        |(None, 180)        |725940     |
|fully_connected_2       |Dense                        |(None, 95)         |17195      |
|dropout                 |Dropout                      |(None, 95)         |0          |
|readout                 |Dense                        |(None, 1)          |96         |
|**Total params**        |                             |                   |**757,331**|
|**Trainable params**    |                             |                   |**757,331**|
|**Non-trainable params**|                             |                   |**0**      |


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.