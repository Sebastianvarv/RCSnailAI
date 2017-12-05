import cv2
import numpy as np
import pandas as pd

def extract_training_data_as_stacked(filename, csv_filename, image_size=(64, 64, 3)):
    """
    Read every 12th frame from input video and bundle every five frames together.

    :param filename:
    :param csv_filename:
    :param image_size:
    :return: images as flattened lists and training labels
    """
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
            frame = frame / 255
            # cv2.imshow("img", frame)
            resized = cv2.resize(frame, image_size[:2])
            processed_frames.append(resized)
            if len(processed_frames) >= 4:
                # cv2.imshow('frame', resized)

                stacked_image = np.concatenate(processed_frames, axis=2)
                # training_labels.append(labels[frame_counter])
                training_images.append(stacked_image)
                # training_images.append(processed_frames.copy())
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


def extract_training_data(filename, csv_filename, image_size=(64, 64, 3)):
    """
    Read every 12th frame from input video and output them.

    :param filename:
    :param csv_filename:
    :param image_size:
    :return: images as flattened lists and training labels
    """
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
            frame = frame / 255
            # cv2.imshow("img", frame)
            resized = cv2.resize(frame, image_size[:2])

            training_images.append(resized)
            training_label_ids.append(frame_counter)

        if cv2.waitKey(1) & 0xFF == ord('q') or not result:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    training_images = np.array(training_images)
    training_labels = labels.loc[training_label_ids]

    assert training_images.shape[0] == training_labels.shape[0]
    return training_images, training_labels