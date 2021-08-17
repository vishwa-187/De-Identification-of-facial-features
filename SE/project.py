from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils
import math

#For detecting a face coordinates should be set in dlib according to the predefined coordinates
#we can easily extract the indexes into the facial landmarks array and extract various facial features
#simply by supplying a string as a key.
facial_features_cordinates = {}
faces_array = []
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))])


#for passing arguments through command line interface 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-i2", "--image2", required=True,
    help="path to input image")
ap.add_argument("-i3", "--image3", required=True,
    help="path to input image")
args = vars(ap.parse_args())


# initialize the list of (x, y)-coordinates
# loop over the 68 facial landmarks and convert them
# to a 2-tuple of (x, y)-coordinates
# return the list of (x, y)-coordinates
def shape_to_numpy_array(shape, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)
    for i in range (0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)
    return coordinates

def visualize_facial_landmarks (image, shape, colors=None, alpha=1):
# create two copies of the input image -- one for the
# overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    # if the colors list is None, initialize it with a black color
    # here we give a color for each facial landmark region in RGB format 
    if colors is None:
        colors = [(0,0,0), (0,0,0), (0,0,0),
                  (0,0,0), (0,0,0),
                  (0,0,0), (0,0,0)]
    # loop over the facial landmark regions individually  
    for (i, name) in enumerate (FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts
        
        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range (1,len(pts)):
                ptA = tuple(pts [l - 1])
                ptB = tuple (pts[l])
                cv2.line (overlay, pta, ptB, colors[i], 2)
        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        #Given a set of points in the plane. the convex hull 
        #of the set is the smallest convex polygon that contains
        #all the points of it.
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
    # apply the transparent overlay        
    cv2.addweighted (overlay, alpha, output, 1 - alpha, 0, output)
    # return the output image
    print (facial_features_cordinates)
    return output


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args ["image"])

image = cv2.resize(image, (300, 300), interpolation = cv2.INTER_AREA)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

image2 = cv2.imread (args ["image2"])
image2 = cv2.resize(image2, (300, 300), interpolation = cv2.INTER_AREA)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)

image3 = cv2.imread(args ["image3"])
image3= cv2.resize(image3, (300,300) , interpolation = cv2.INTER_AREA)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
faces3 = face_cascade.detectMultiScale(gray3, 1.1, 4)

faces2 = faces2
faces3 = faces3
pointsf1=[]
pointsf2=[]
pointsf3=[]

for (x, y, w, h) in faces:
    for i in range (x,x+w):
        for j in range (y, y+h):
            pointsf1.append([i,j])
    cv2.rectangle (image, (x, y), (x+w, y+h), (255, 0, 0), 2)
for (x, y, w, h) in faces2:
    for i in range (x,x+w):
        for j in range (y, y+h) :
            pointsf2.append([i,j])
    cv2.rectangle (image2, (x, y), (x+w, y+h), (255, 0, 0), 2)
for (x, y, w, h) in faces3:
    for i in range (x,x+w):
        for j in range (y, y+h):
            pointsf3.append([i,j])
    cv2.rectangle (image3, (x, y), (x+w, y+h), (255, 0, 0), 2)
for i in range (0,300):
    for j in range (0,300):
        for j in range (y, y+h):
            pointsf2.append([i,j])
    cv2.rectangle(image2, (x, y), (x+w, y+h), (255, 0, 0), 2)
for (x, y, w, h) in faces3:
    for i in range (x,x+w):
        for j in range (y, y+h):
            pointsf3.append([i,j])
    cv2.rectangle (image3, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    
for i in range (0,300):
    for j in range (0,300):
        a=math.pow(gray[i][j],2)
        b=math.pow(gray2[i][j],2)
        y=np.add(a,b)
        z=math.sqrt(y)
        gray2[i][j] = np.divide(z,2).astype(int)
        
cv2.imshow("image", gray);
cv2.waitKey(0);

cv2.imshow("image3", gray3);
cv2.waitKey(0);

cv2.imshow("image2", gray2);
cv2.waitKey(0);

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)
    output = visualize_facial_landmarks(image, shape)
    #cv2.imshow("Image", output)
    cv2.waitKey(0)



