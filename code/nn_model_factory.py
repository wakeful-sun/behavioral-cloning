from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Conv2D, MaxPooling2D, Dropout, Lambda


def create_model(weights_path=None):
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(((70, 25), (0, 0))))
    model.add(Conv2D(10, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(22, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(36, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(180))
    model.add(Dense(106))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    if weights_path:
        model.load_weights(weights_path)

    return model
