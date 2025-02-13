from fastapi import FastAPI, HTTPException
import cv2
import numpy as np
import requests
from fingerprint_enhancer import enhance_fingerprint
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from typing import List

app = FastAPI()

def download_image(url):
    """Download an image from a URL and convert it to grayscale."""
    try:
        print(f"Downloading image from: {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to download image: {url}, Status Code: {response.status_code}, Response: {response.text}")
            return None

        img = Image.open(BytesIO(response.content)).convert("L")  # Convert to grayscale
        img = np.array(img, dtype=np.uint8)  # Ensure uint8 format for OpenCV
        return img
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def process_fingerprint(image):
    """Enhance the fingerprint for better feature detection."""
    try:
        print("Enhancing fingerprint image...")  # Debugging log
        # Ensure the image is in uint8 format
        if image.dtype != np.uint8:
            image = np.uint8(image)

        # Apply enhancement (assumes enhance_fingerprint returns a proper numpy array)
        # return enhance_fingerprint(image)
        return image
    except Exception as e:
        print(f"Error enhancing fingerprint: {e}")
        return None

def compare_fingerprints(img1, img2):
    """Compare fingerprints using ORB feature matching."""
    try:
        print("Comparing fingerprints...")  # Debugging log
        print(f"Image 1 dtype: {img1.dtype}, Image 2 dtype: {img2.dtype}")  # Debug image types
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            print("One of the fingerprints does not have enough keypoints")
            return False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        match_result = len(matches) > 20  # Adjust threshold based on testing
        print(f"Number of matches found: {len(matches)}, Match Result: {match_result}")
        return match_result
    except Exception as e:
        print(f"Error comparing fingerprints: {e}")
        return False

class StoredFingerprint(BaseModel):
    url: str
    candidateId: int  # Ensure candidateId is an integer

class FingerprintComparisonRequest(BaseModel):
    new_fingerprint_url: str
    stored_fingerprints: List[StoredFingerprint]

@app.post("/compare/")
async def compare_fingerprints_api(data: FingerprintComparisonRequest):
    try:
        new_fingerprint = download_image(data.new_fingerprint_url)
        if new_fingerprint is None:
            raise HTTPException(status_code=400, detail="Invalid new fingerprint image")

        new_fingerprint = process_fingerprint(new_fingerprint)

        for fingerprint in data.stored_fingerprints:
            stored_img = download_image(fingerprint.url)
            if stored_img is None:
                continue

            stored_img = process_fingerprint(stored_img)

            if compare_fingerprints(new_fingerprint, stored_img):
                return {"match": True, "candidateId": fingerprint.candidateId}

        return {"match": False}
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": str(e)}