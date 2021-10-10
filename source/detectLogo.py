from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os
import pickle

# Get Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(ROOT_DIR)
# testPath = head + "/" + "Test"
TrainedModel = loaded_model = pickle.load(open("Models/TrainedModel.yml", 'rb'))

# Init Lists
hists = []  # histogram of Image
labels = []  # Label of Image


def detect(image):
    # Convert to Gray and Resize
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    md = np.median(gray)
    sigma = 0.35
    low = int(max(0, (1.0 - sigma) * md))
    up = int(min(255, (1.0 + sigma) * md))
    # Create Edged Image from Gray Scale
    edged = cv.Canny(gray, low, up)
    # cv.imwrite("edge.jpg", edged)

    # extract only shape in image
    (x, y, w, h) = cv.boundingRect(edged)
    logo = gray[y:y + h, x:x + w]
    logo = cv.resize(gray, (200, 100))

    # Calculate Histogram of Test Image
    hist = feature.hog(
        logo,
        orientations=9,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L1"
    )
    # Predict in model
    predict = TrainedModel.predict(hist.reshape(1, -1))[0]

    logoName = predict.title()

    return logoName


cam = cv.VideoCapture(0)
while True:
    success, img = cam.read()
    originalImage = img.copy()
    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    title = detect(img)
    cv.putText(img, title, (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv.imshow('image', img)
    cv.waitKey(1)
