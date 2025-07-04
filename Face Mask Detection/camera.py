from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
# from google.colab import drive

def DetectMask(frame, faceNet, maskNet):
    # This function takes a video frame, the face detection model (faceNet),
    # and the mask detection model (maskNet) as input.
    # It returns the locations of detected faces and the mask predictions for each face.
    # grab the dimensions of the frame and then construct a blob
    # from the frame
    (h, w) = frame.shape[:2]
    # Create a blob from the image for face detection.
    # A blob is a 4D array object for neural networks.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
    (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections
    faceNet.setInput(blob)
    detection = faceNet.forward()
    print(detection.shape)

    # initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
    faces = [] # List to store the detected face images (preprocessed)
    locs = [] # List to store the bounding box locations of the faces
    pred = [] # List to store the mask predictions for each face

    for i in range(0, detection.shape[2]):
        # extract the confidence (probability) associated with the detection
        confidence = detection[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype = "float32")
        pred = maskNet.predict(faces, batch_size = 32)
    return (locs, pred)

# Determine the base path for model files.
# In a local environment, this gets the directory of the current script.
# In Colab, this might need adjustment depending on how files are accessed.
base_path = os.path.dirname(os.path.abspath(__file__))
prototxtPath = os.path.join(base_path, "face_detector", "deploy.prototxt")
weightsPath = os.path.join(base_path, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
#this model was saved in .h5 format so if you have a different format you'll need to change it
maskNet = load_model(os.path.join(base_path, "mask_detection.h5"))

print("Model files:")
print(f"Checking prototxt path: {prototxtPath} (exists: {os.path.exists(prototxtPath)})")
print(f"Checking weights path: {weightsPath} (exists: {os.path.exists(weightsPath)})")

print("starting camera...")

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) = DetectMask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Face Mask Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()