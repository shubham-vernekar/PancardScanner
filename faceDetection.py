import cv2
import numpy as np
from glob import glob
import requests
import base64
import json
import pytesseract
import re

inputFolder = "Images"

# Make this True if you want to use Google's Vision cloud API which is more robust than Tesseract OCR (open-source).
# To use vision API you will need an API key https://cloud.google.com/vision/
useVisionAPI = False
visionAPIKey = "ENTER_VISION_API_KEY_HERE"

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

def performOCR(image,threshold=0):
    """ Extract text from image. Filter the results according to the thresholds (Used when using tesseract) """

    if useVisionAPI:

        # Convert image to base64 format
        base64Image = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')

        jsonObject = {
            "requests": [
                {
                "image": {
                    "content": base64Image
                },
                "features": [
                    {
                    "type": "DOCUMENT_TEXT_DETECTION"
                    }
                ]
                }
            ]
        }

        url = "https://vision.googleapis.com/v1/images:annotate?key={}".format(visionAPIKey) 
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        r = requests.post(url, data=json.dumps(jsonObject), headers=headers).json()

        # Find the ocr text from the output JSON
        extractedData = r.get("responses",[{}])[0].get("fullTextAnnotation",{}).get("text","")
    else:
        extractedData = ""

    if extractedData.strip()!="":
        return extractedData
    else:

        # Using pytesseract extract text from image. Filter the results according to the thresholds 
        data = pytesseract.image_to_data(image, config='--psm 6 --oem 1', output_type= pytesseract.Output.STRING)
        data = data.split("\n")
        response = ""

        # Parse the tesseract output
        for record in data[1:]:
            record = record.split("\t")
            if len(record)==12:
                if record[11].strip()=="":
                    response+="\n"
                elif int(record[10]) >= threshold:
                    response+=record[11]+" "

        return response

def cropImage(img,rect):
    """ Crop image after adding constraints to make sure they dont go out of bounds"""
    h,w,_ = img.shape
    y1,y2,x1,x2 = rect
    arr = [0 if x<0 else x for x in [x1,x2,y1,y2]]
    x1,x2,y1,y2 = [w-1 if x>w else x for x in arr[:2]]+[h-1 if x>h else x for x in arr[2:]]
    return img[y1:y2,x1:x2]


face_cascade = cv2.CascadeClassifier(
    "Models/haarcascade_frontalface_default.xml")

filesList = glob(inputFolder+"/*.jpg") + glob(inputFolder+"/*.png")


for filePath in filesList:
    pancardNumber = ""
    image = cv2.imread(filePath)
    image = image_resize(image, 500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    x,y,w,h = faces[0]

    # Extract Face
    delta = 0.3   # Add buffer to extract the entire face 
    face = image[y-int(delta*h):y+h+int(delta*h), x-int(delta*w):x+w+int(delta*w)]

    # Extract Pan Number
    panROI = cropImage(image,(y-int(h*0.5),y+int(h*0.8),0,x-int(h*0.8)))
    pancardNumber = performOCR(panROI,threshold=20)
    # Remove noise
    regexMatch = re.search(r"\bpermanant\b[^\n]*",pancardNumber,flags=re.IGNORECASE)
    if not regexMatch:
        regexMatch = re.search(r"\baccount\b[^\n]*",pancardNumber,flags=re.IGNORECASE)
    if not regexMatch:
        regexMatch = re.search(r"\bnumber\b[^\n]*",pancardNumber,flags=re.IGNORECASE)

    if regexMatch:
        pancardNumber = pancardNumber[regexMatch.end():].strip()

        regexMatch = re.search(r"[^a-z0-9 ]",pancardNumber,flags=re.IGNORECASE)
        if regexMatch:
            pancardNumber = pancardNumber[:regexMatch.start()].strip()

    print (pancardNumber)

    # cv2.imshow("img",panROI)
    # cv2.waitKey(0)
    # quit()
    