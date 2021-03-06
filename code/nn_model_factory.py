from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Conv2D, MaxPooling2D, Dropout, Lambda


def create_model(dropout):
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3), name="normalization"))
    model.add(Cropping2D(((70, 25), (0, 0)), name="cropping"))
    model.add(Conv2D(6, (5, 5), activation="relu", name="conv1_5x5_relu"))
    model.add(MaxPooling2D(name="max_pooling1_2x2"))
    model.add(Conv2D(16, (5, 5), activation="relu", name="conv2_5x5_relu"))
    model.add(MaxPooling2D(name="max_pooling2_2x2"))
    model.add(Conv2D(28, (5, 5), activation="relu", name="conv3_5x5_relu"))
    model.add(MaxPooling2D(name="max_pooling3_2x2"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(180, name="fully_connected_1"))
    model.add(Dense(95, name="fully_connected_2"))
    model.add(Dropout(dropout, name="dropout"))
    model.add(Dense(1, name="readout"))

    return model, "LeNet with extended dense layer"
