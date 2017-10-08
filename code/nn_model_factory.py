from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Conv2D, MaxPooling2D, Dropout, Lambda


def create_model(weights_path=None):
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(((70, 25), (0, 0))))
    model.add(Conv2D(6, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(28, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    if weights_path:
        model.load_weights(weights_path)

    return model
