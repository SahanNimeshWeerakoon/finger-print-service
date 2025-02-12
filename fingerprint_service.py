from fastapi import FastAPI, HTTPException
import cv2
import numpy as np
import requests
from fingerprint_enhancer import enhance_fingerprint
from PIL import Image
from io import BytesIO

app = FastAPI()

def download_image(url):
    """Download an image from a URL and convert it to grayscale."""
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None
        img = Image.open(BytesIO(response.content)).convert("L")  # Convert to grayscale
        return np.array(img)
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def process_fingerprint(image):
    """Enhance the fingerprint for better feature detection."""
    return enhance_fingerprint(image)

def compare_fingerprints(img1, img2):
    """Compare fingerprints using ORB feature matching."""
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    return len(matches) > 20  # Adjust threshold based on testing

@app.post("/compare/")
async def compare_fingerprints_api(new_fingerprint_url: str, stored_fingerprints: list):
    """
    Compare a new fingerprint image with stored fingerprint images.
    stored_fingerprints: List of dictionaries with 'url' and 'candidateId'.
    """
    new_fingerprint = download_image(new_fingerprint_url)
    if new_fingerprint is None:
        raise HTTPException(status_code=400, detail="Invalid new fingerprint image")

    new_fingerprint = process_fingerprint(new_fingerprint)

    for fingerprint in stored_fingerprints:
        stored_img = download_image(fingerprint["url"])
        if stored_img is None:
            continue

        stored_img = process_fingerprint(stored_img)

        if compare_fingerprints(new_fingerprint, stored_img):
            return {"match": True, "candidateId": fingerprint["candidateId"]}

    return {"match": False}