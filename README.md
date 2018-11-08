# PancardScanner
Extracts information such as Pan number, Name and Data of Birth from photos of Pan Cards.
The application uses face detection to extract the photo from pancard and uses OCR to extract text information from the pan card image

## Setup Instructions

### Prerequisites

<a href = "https://github.com/tesseract-ocr/tesseract" >Tesseract OCR </a> OR <a href = "https://cloud.google.com/vision/" > Google vision API </a> key  

### Install Dependencies

```bash
$ pip install -r requirements.txt
# This will install all dependencies
```
#### Install Tesseract
https://github.com/tesseract-ocr/tesseract/wiki

### How to run

```bash
$ python wrapper.py
```

To import this code in python
```python
from extractData import extract_pancard_data

filePath = "test.jpg"

# Make this True if you want to use Google's Vision cloud API which is more robust than Tesseract OCR (open-source).
# To use vision API you will need an API key https://cloud.google.com/vision/
useVisionAPI = False
visionAPIKey = "ENTER_VISION_API_KEY_HERE"

pancardNumber, name, dob, face = extract_pancard_data(
        filePath, useVisionAPI=useVisionAPI, visionAPIKey=visionAPIKey)

```
