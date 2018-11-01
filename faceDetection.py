import cv2
import numpy as np
from glob import glob

inputFolder = "Images"

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """  Resize image keeping same aspect ratio and specific height or width """

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



face_cascade = cv2.CascadeClassifier(
    "Models/haarcascade_frontalface_default.xml")

filesList = glob(inputFolder+"/*.jpg") + glob(inputFolder+"/*.png")



for filePath in filesList:
    image = cv2.imread(filePath)
    image = image_resize(image, 500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    x,y,w,h = faces[0]

    # Extract Face
    delta = 0.3   # Add buffer to extract the entire face 
    face = image[y-int(delta*h):y+h+int(delta*h), x-int(delta*w):x+w+int(delta*w)]

