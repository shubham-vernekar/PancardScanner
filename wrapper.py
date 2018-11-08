from extractData import extract_pancard_data
from glob import glob
import cv2

inputFolder = r"Images"

# Make this True if you want to use Google's Vision cloud API which is more robust than Tesseract OCR (open-source).
# To use vision API you will need an API key https://cloud.google.com/vision/
useVisionAPI = False
visionAPIKey = "ENTER_VISION_API_KEY_HERE"

# Read all images in the folder
filesList = glob(inputFolder+"/*.jpg") + glob(inputFolder+"/*.png")

for filePath in filesList:

    pancardNumber, name, dob, face = extract_pancard_data(
        filePath, useVisionAPI=useVisionAPI, visionAPIKey=visionAPIKey)

    print("Pancard Number: ", pancardNumber)
    print("Name: ", name)
    print("DOB: ", dob)
    print("----------------------------")

    cv2.imshow("img", face)
    cv2.waitKey(0)
