from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os
import pickle
from sklearn import metrics


# Get Paths


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(ROOT_DIR)
trainingPath = head + "/" + "Train"
testPath = head + "/" + "Test"
TrainedModel = loaded_model = pickle.load(open("Models/TrainedModel.yml", 'rb'))

# Init Lists
hists = []  # histogram of Image
Test_labels = []  # Label of Image
#actual_labels = []

#get actual data labels
#for imagePath in glob.glob(trainingPath + "/*/"):
   # actual = imagePath.split("/")[-2]
    #actual_labels.append(actual)
#print(*actual_labels)
#print(len(actual_labels))

# Check Test Images for Model
for (imagePath) in glob.glob(testPath + "/*.*"):
    # Read Images
    image = cv.imread(imagePath)
    try:
        # Convert to Gray and Resize
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
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

        # Make pictures default Height
        height, width = image.shape[:2]
        reWidth = int((300 / height) * width)
        image = cv.resize(image, (reWidth, 300))

        # Write predicted label over the Image
        pred_title = predict.title()
        Test_labels.append(pred_title)
        cv.putText(image,pred_title , (10, 30), cv.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 4)

        # Get Image name and show Image
        imageName = imagePath.split("/")[-1]

        cv.imshow(imageName, image)
        cv.waitKey(0)
        # Close Image
        cv.destroyAllWindows()
    except cv.error:
        # If Image couldn't be Read
        print(imagePath)
        print("Test Image couldn't be read")

#print(metrics.confusion_matrix(actual_labels, Test_labels , labels=None))
# Printing the precision and recall, among other metrics
#print(metrics.classification_report(actual_labels, Test_labels , labels=None))