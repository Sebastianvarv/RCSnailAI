import cv2
import numpy as np
import pandas as pd
from keras.layers import Conv2D, Flatten, Dense, LSTM, TimeDistributed, \
    MaxPooling2D
from keras.models import Sequential

IMAGE_SIZE = (64, 64, 3)

def extract_training_data(filename, csv_filename, image_size=(64, 64, 3)):
    cap = cv2.VideoCapture(filename)
    labels = pd.read_csv(csv_filename, sep="\t")

    frame_counter = 0
    processed_frames = []

    training_images = []
    training_label_ids = []
    while True:
        frame_counter += 1
        result, frame = cap.read()
        if result and frame_counter % 12 == 0:
            # cv2.imshow("img", frame)
            resized = cv2.resize(frame, image_size[:2])
            processed_frames.append(resized)
            if len(processed_frames) >= 4:
                # cv2.imshow('frame', resized)

                # stacked_image = np.concatenate(processed_frames, axis=2)
                # training_labels.append(labels[frame_counter])
                training_images.append(processed_frames.copy())
                training_label_ids.append(frame_counter)
                processed_frames.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q') or not result:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    training_images = np.array(training_images)
    training_labels = labels.loc[training_label_ids]

    assert training_images.shape[0] == training_labels.shape[0]
    return training_images, training_labels


if __name__ == "__main__":
    images, labels = extract_training_data("Data/20171029-201949.h264.avi", "Data/20171029-201949.h264.csv", IMAGE_SIZE)

    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), kernel_initializer="he_normal", activation='relu'),
                              input_shape=(4, 64, 64, 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(4, activation='linear'))

    model.compile(loss="mse", optimizer="adam")
    model.summary()

    print(labels.head())

    y_train = labels.as_matrix(columns=labels.columns[1:])

    prediction = model.predict(images)

    for i in range(len(prediction)):
        if i % 10 == 0:
            print(np.mean(images[i], axis=(1, 2, 3)), prediction[i])

    # for i in range(len(images)):
    #     # print(np.mean(images[i], axis=(1,2,3)))
    #     print(labels.head(10))

    history = model.fit(images, y_train, batch_size=64, epochs=5, validation_split=0.04)

    images, _ = extract_training_data("Data/20171029-201639.h264.avi", "Data/20171029-201639.h264.csv", IMAGE_SIZE)

    prediction = model.predict(images)

    for i in range(len(prediction)):
        if i % 10 == 0:
            print(np.mean(images[i], axis=(1,2,3)), prediction[i])

    # #
    # for i in range(len(images)):
    #     print(np.mean(images[i], axis=(1,2,3)))
