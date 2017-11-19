import cv2
import pandas as pd
import numpy as np

cap = cv2.VideoCapture("Data/20171029-201639.h264.avi")
labels = pd.read_csv("Data/20171029-201639.h264.csv")

frame_counter = 0
processed_frames = []
IMAGE_SIZE = (64, 64, 3)


training_images = []
while True:
    frame_counter += 1
    result, frame = cap.read()
    if result and frame_counter % 12 == 0:
        resized = cv2.resize(frame, IMAGE_SIZE[:2])
        processed_frames.append(resized)
        if len(processed_frames) >= 4:
            # cv2.imshow('frame', resized)

            stacked_image = np.concatenate(processed_frames, axis=2)
            training_images.append(stacked_image)
            processed_frames.pop()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

training_images = np.asarray(training_images)

print(training_images.shape)