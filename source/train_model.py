from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os
import pickle

# Get Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(ROOT_DIR)
trainingPath = head + "/" + "logos"

# Init Lists
hists = []  # histogram of Image
labels = []  # Label of Image

for imagePath in glob.glob(trainingPath + "/*/*.*"):
    # get label from folder name
    label = imagePath.split("/")[-2]
    #print(label)
    image = cv.imread(imagePath)
    try:
        # RGB to Gray
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Calculate Low and Up value to extract Edges
        md = np.median(gray)
        sigma = 0.35
        low = int(max(0, (1.0 - sigma) * md))
        up = int(min(255, (1.0 + sigma) * md))
        # Create Edged Image from Gray Scale
        edged = cv.Canny(gray, low, up)

        # extract only shape in image
        (x, y, w, h) = cv.boundingRect(edged)
        logo = gray[y:y + h, x:x + w]
        logo = cv.resize(logo, (200, 100))

        # Calculate histogram
        hist = feature.hog(
            logo,
            orientations=9,
            pixels_per_cell=(10, 10),
            cells_per_block=(2, 2),
            transform_sqrt=True,
            block_norm="L1"
        )
        # Add value into Lists
        hists.append(hist)
        labels.append(label)
    except cv.error:
        # If Image couldn't be Read
        print(imagePath)
        print("Training Image couldn't be read")

with open("pickels/labels.pickle", 'wb') as f:
    pickle.dump(labels, f)

# Create model as Nearest Neighbors Classifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(hists, labels)

# create a directory to save the model
os.makedirs(os.path.dirname("Models/TrainedModel.yml"), exist_ok=True)
# save the model using pickle
pickle.dump(model, open("Models/TrainedModel.yml", 'wb'))
