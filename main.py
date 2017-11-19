import cv2
import pandas as pd

cap = cv2.VideoCapture("Data/20171029-201639.h264.avi")
labels = pd.read_csv("20171029-201639.h264.csv")


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    resized = cv2.resize(frame, (64, 64))

    cv2.imshow('frame', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()