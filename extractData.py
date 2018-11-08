import cv2
import numpy as np
import requests
import base64
import json
import pytesseract
import re


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def perform_OCR(image, useVisionAPI, visionAPIKey, threshold=0):
    """ Extract text from image. Filter the results according to the thresholds (Used when using tesseract) """

    if useVisionAPI:

        # Convert image to base64 format
        base64Image = base64.b64encode(
            cv2.imencode('.jpg', image)[1]).decode('utf-8')

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

        url = "https://vision.googleapis.com/v1/images:annotate?key={}".format(
            visionAPIKey)
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        r = requests.post(url, data=json.dumps(
            jsonObject), headers=headers).json()

        # Find the ocr text from the output JSON
        extractedData = r.get("responses", [{}])[0].get(
            "fullTextAnnotation", {}).get("text", "")
    else:
        extractedData = ""

    if extractedData.strip() != "":
        return extractedData
    else:

        # Using pytesseract extract text from image. Filter the results according to the thresholds
        data = pytesseract.image_to_data(
            image, config='--psm 6 --oem 1', output_type=pytesseract.Output.STRING)
        data = data.split("\n")
        response = ""

        # Parse the tesseract output
        for record in data[1:]:
            record = record.split("\t")
            if len(record) == 12:
                if record[11].strip() == "":
                    response += "\n"
                elif int(record[10]) >= threshold:
                    response += record[11]+" "

        return response


def crop_image(img, rect):
    """ Crop image after adding constraints to make sure they dont go out of bounds"""
    h, w, _ = img.shape
    y1, y2, x1, x2 = rect
    arr = [0 if x < 0 else x for x in [x1, x2, y1, y2]]
    x1, x2, y1, y2 = [w-1 if x > w else x for x in arr[:2]] + \
        [h-1 if x > h else x for x in arr[2:]]
    return img[y1:y2, x1:x2]


def find_face(image):
    """ Finds human faces in the image """

    faceCascade = cv2.CascadeClassifier(
        "Models/haarcascade_frontalface_default.xml")

    eyeCascade = cv2.CascadeClassifier(
        "Models/haarcascade_eye.xml")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return []

    if len(faces) > 1:
        # More than One face found. Likely one is a false positive
        # Check for eyes in the face
        for face in faces:
            x, y, w, h = face
            roi = gray[y:y+h, x:x+w]
            eyes = eyeCascade.detectMultiScale(roi)
            if len(eyes) > 0:
                # If eyes are found return face
                return face

    return faces[0]


def extract_pancard_data(filePath, useVisionAPI=False, visionAPIKey=""):
    """ Extact details from pancard. Returns Pancard number, Name, Date of Birth and Photo of the face """

    image = cv2.imread(filePath)

    image = image_resize(image, 500)
    face = find_face(image)

    if len(face) == 0:
        return None, "", "", ""
    else:
        x, y, w, h = face

    # Extract Face
    x, y, w, h = face
    # image = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    delta = 0.3   # Add buffer to extract the entire face
    faceROI = image[y-int(delta*h):y+h+int(delta*h), x -
                    int(delta*w):x+w+int(delta*w)]

    # Extract Pan Number
    panROI = crop_image(image, (y-int(h*0.5), y+int(h*0.8), 0, x-int(h*0.8)))
    pancardNumber = perform_OCR(
        panROI, useVisionAPI, visionAPIKey, threshold=20)
    # Remove noise
    regexMatch = re.search(
        r"\bpermanant\b[^\n]*", pancardNumber, flags=re.IGNORECASE)
    if not regexMatch:
        regexMatch = re.search(
            r"\baccount\b[^\n]*", pancardNumber, flags=re.IGNORECASE)
    if not regexMatch:
        regexMatch = re.search(
            r"\bnumber\b[^\n]*", pancardNumber, flags=re.IGNORECASE)

    if regexMatch:
        pancardNumber = pancardNumber[regexMatch.end():].strip()

        regexMatch = re.search(
            r"[^a-z0-9 ]", pancardNumber, flags=re.IGNORECASE)
        if regexMatch:
            pancardNumber = pancardNumber[:regexMatch.start()].strip()

    # Extract Name
    nameROI = crop_image(image, (y-int(h*2.1), y-int(h*0.7), 0, x-int(h*0.8)))
    name = perform_OCR(nameROI, useVisionAPI, visionAPIKey, threshold=20)
    regexMatch = re.search(r"\bincome\b[^\n]*", name, flags=re.IGNORECASE)
    if not regexMatch:
        regexMatch = re.search(r"\btax\b[^\n]*", name, flags=re.IGNORECASE)
    if not regexMatch:
        regexMatch = re.search(
            r"\bdepartment\b[^\n]*", name, flags=re.IGNORECASE)

    if regexMatch:
        name = name[regexMatch.end():].strip()

    name = re.sub(r" +", " ", name).strip()

    name = name.split("\n")
    name = [x for x in name if len(x.replace(" ", "")) > 4][0]

    # Extract DOB
    dobROI = crop_image(image, (y-int(h*1.5), y+int(h*0.8), 0, x-int(h*0.8)))
    dob = perform_OCR(dobROI, useVisionAPI, visionAPIKey, threshold=20)
    dob = re.search(r'(\d+ */\d+/ *\d+)', dob, flags=re.IGNORECASE).group()

    return pancardNumber, name, dob, faceROI
