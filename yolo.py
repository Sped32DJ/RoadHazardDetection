#!/usr/bin/env python3
"""Run YOLOv8 object detection on an image and print results.
Usage:
  python miniYolo.py /path/to/image.jpg
"""

import argparse
import os
import sys
# To encode EXIF data and images
import base64
from io import BytesIO
from datetime import datetime
# Our YOLO model
from ultralytics import YOLO
import cv2

# Out database import
from firebase_admin import credentials, firestore, initialize_app, get_app
# Image croppping and EXIF data extraction imports
from PIL import Image
import piexif

def parse_args():
	p = argparse.ArgumentParser(description="Run YOLO on an image or video file")
	p.add_argument('source', nargs='?', help='Path to image/video file. If omitted you will be prompted.')
	p.add_argument('-m', '--model', default='models/best.pt', help='Path to YOLO model file (default: models/best.pt)')
	p.add_argument('--show', action='store_true', help='Show results (opens image/window)')
	p.add_argument('--no-save', action='store_true', help='Do not save annotated output files')
	p.add_argument('--project', default='runs/detect', help='Output base directory (default: runs/detect)')
	p.add_argument('--name', default='predict', help='Output run name/folder (default: predict)')
	
	# Firestore options
	p.add_argument('--firestore', action='store_true', help='Send detection data to Google Firestore')
	p.add_argument('--creds', default='firebase-credentials.json', help='Path to Firebase credentials JSON file')
	p.add_argument('--crop-images', action='store_true', help='Crop and upload detected pothole images to Firestore')
	p.add_argument('--collection', default='detections', help='Firestore collection name (default: detections)')
	return p.parse_args()

# Function to extract GPS data from EXIF
def getGPSfromExif(imagePath):
    """Extract GPS coordinates from EXIF data.
    Returns a dict with 'latitude', 'longitude', and 'altitude' (if available), or None if GPS data is not found.
    """
    #gps_ifd = imagePath.get("GPS")
    exifDict = piexif.load(imagePath)
    gps_ifd = exifDict.get("GPS")
    if not gps_ifd:
        return None

    try:
        exifDict = piexif.load(imagePath)
        gps_ifd = exifDict.get("GPS")
        if not gps_ifd:
            return None
        def ConvToDegrees(value):
            """Convert EXIF GPS coordinates to decimal degrees."""
            d, m, s = value
            return d[0] / d[1] + (m[0] / m[1]) / 60.0  + (s[0] / s[1] / 3600.0)

        # Calc lat and long in decimal degrees
        lat = ConvToDegrees(gps_ifd[piexif.GPSIFD.GPSLatitude])
        lon = ConvToDegrees(gps_ifd[piexif.GPSIFD.GPSLongitude])
        # Check directions of lat and long
        lat_ref = gps_ifd[piexif.GPSIFD.GPSLatitudeRef].decode()
        lon_ref = gps_ifd[piexif.GPSIFD.GPSLongitudeRef].decode()
        if lat_ref == "S":
            lat = -lat
        if lon_ref == "W":
            lon = -lon

        # Extract altitude (if available)
        altitude = None
        if piexif.GPSIFD.GPSAltitude in gps_ifd:
            alt = gps_ifd[piexif.GPSIFD.GPSAltitude]
            altitude = alt[0] / alt[1]

        return {
            "latitude": round(lat, 8),
            "longitude": round(lon, 8),
            "altitude": round(altitude, 2) if altitude is not None else None,
        }
    except Exception as e:
        print(f"Error extracting GPS data: {e}")
        return None


#Initialize Firestore client
def init_firestore(creds_path):
    """Initialize Firestore client using provided credentials."""
    if not os.path.exists(creds_path):
        print(f"Error: Firebase credentials file not found: {creds_path}")
        return None

    try:
        # Check if Firebase app is already initialized
        cred = credentials.Certificate(creds_path)
        try:
            get_app()
        except ValueError:
            initialize_app(cred)
        db = firestore.client()
        print("Firestore initialized successfully.")
        return db
    except Exception as e:
        print(f"Error initializing Firestore: {e}")
        return None

# Crop detected pothole from image 
def crop_pothole(image, bbox, padding=10):
    """Crop the detected pothole from the image using the bounding box.
    Args:
        image (numpy array): OpenCV format image (BGR).
        bbox [x1, y1, x2, y2]: Bounding box coordinates.
        padding (int): padding around the bounding box in pixels.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        return image[y1:y2, x1:x2]
    except Exception as e:
        print(f"Error cropping pothole image: {e}")
        return None

# Convert OpenCV image to base64 string for Firestore upload
def imageToBase64(image):
    """Convert OpenCV image to base64 string."""
    try:
        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Compress image to reduce size (reduce memory usage and upload time)
        pil_image.thumbnail((400,400))

        # Save to bytes buffer and encode as base64
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=70)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_base64
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

# Upload the detection to firestore
def uploadDetectionToFirestore(db, detectionData, collection='detections'):
    """Upload detection data to Firestore.
    Args:
        db: Firestore client instance.
        detectionData: dict containing detection information (e.g., GPS, timestamp, image).
        collection: Firestore collection name to store the detection (default: 'detections').
    Returns:
        The ID of the created Firestore document, or None if upload failed.
    """

    try:
        # Create a new document in the specified collection with the detection data
        doc_ref = db.collection(collection).document()
        doc_ref.set(detectionData)
        return doc_ref.id
    except Exception as e:
        print(f"Error uploading detection to Firestore: {e}")
        return None

# Resolve class ID to human-readable label using the model's names list
def resolveLabel(names, classID):
    """Resolve class ID to human-readable label using the model's names list."""

    # Handle case where names is a dict and classID is an integer
    if isinstance(names, dict):
        return names.get((classID), str(classID))

    # Handle case where names is a list and classID is an index
    if isinstance(names, (list, tuple)) and 0 <= classID < len(names):
        return str(names[classID])
    return str(classID)

# Print detection results and optionally upload to Firestore
def printDetectionResults(results, model, source_file=None, db=None, args=None):
    """Print detection results and optionally upload to Firestore."""
    names = {}
    try:
        names = model.model.names
    except Exception as e:
        print(f"Error accessing model names: {e}")

    # Loop through each result and print detections
    for idx, res in enumerate(results):
        print(f"Result {idx}:")
        boxes = getattr(res, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            print("  No detections found.")
            continue
        
        # boxes.cls, boxes.conf, boxes.xyxy are tensors
        try:
            xyxy = boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            conf = boxes.conf.cpu().numpy()    # Confidence scores
            cls = boxes.cls.cpu().numpy()      # Class IDs
        except Exception as e:
            print(f"  Error processing detection results: {e}")
            continue

        # Get source image for cropping and GPS extraction
        source_image = getattr(res, 'orig_img', None)
        gps_data = None
        if source_file and os.path.isfile(source_file):
            # Try to extract GPS data from the source image's EXIF metadatat
            # NOTE: filetype is a limitation here - we can only extract GPS data from image files that support EXIF metadata
            if(source_file.lower().endswith(('.jpg', '.jpeg', '.png'))):
                gps_data = getGPSfromExif(source_file)
                if gps_data:
                    print(f"  GPS data: {gps_data}")
                elif source_image is None:
                    source_image = cv2.imread(source_file)  # Load image with OpenCV if not already loadedv
        
        # Process each detection
        for i in range(len(conf)):
            classID = int(cls[i])
            label = resolveLabel(names, classID)
            confidence = float(conf[i])
            bbox = xyxy[i]  # Bounding box coordinates for this detection
            print(f"  Detection {i+1}: {label} (confidence: {confidence:.2f}) bbox={bbox}")

            # Upload to Firestore  if enabled and confidence is above threshold (0.5)
            if db and args and confidence > 0.5:
                detectionData = {
                    "timestamp": datetime.now(),
                    "source_file": source_file or 'unknown',
                    "label": label,
                    "class_id": classID,
                    "confidence": float(confidence),
                    "gps": gps_data or None
                }

                # If cropping is enabled and we have the source image, crop the detected pothole and include it in the upload
                if args.crop_images and source_image is not None:
                    cropped_image = crop_pothole(source_image, bbox)
                    if cropped_image is not None:
                        img_base64 = imageToBase64(cropped_image)
                        if img_base64:
                            detectionData['cropped_image'] = img_base64
                            detectionData['image_size'] = cropped_image.shape[:2]  # (height, width)

                doc_id = uploadDetectionToFirestore(db, detectionData, collection=args.collection)
                if doc_id:
                    print(f"    Uploaded to Firestore with document ID: {doc_id}")
                else:
                    print("    Failed to upload to Firestore.")


def main():
    args = parse_args()

    source = args.source

    if not source:
        source = input("Enter path to image/video file: ").strip()
        if not source:
            print("Error: No input source provided. Exiting.")
            sys.exit(1)
    
    if not source:
        print("Error: No input source provided. Exiting.")
        sys.exit(1)
    
    if not os.path.exists(source):
        print(f"Error: file not found: {source}", file=sys.stderr)
        sys.exit(1)
    
    #Load model
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    save_outputs = not args.no_save
    try:
        results = model.predict(
            source=source,
            save=save_outputs,
            project=args.project,
            name=args.name,
            exist_ok=True
        )
    except Exception as e:
        print(f"Error running inference (detection): {e}", file=sys.stderr)
        sys.exit(1)

    #Print detections to stdout
    db = None
    if args.firestore:
        db = init_firestore(args.creds)
        if not db:
            print("Error: Firestore initialization failed. Detection data will not be uploaded.", file=sys.stderr)

    printDetectionResults(results, model, source_file=source, db=db, args=args)

    # Display optional result window if --show is enabled
    if args.show:
        for r in results:
            try:
                r.show()
            except Exception as e:
                print(f"Error displaying results: {e}")
                continue

    # Print path to saved annotated output if saving is enabled
    if save_outputs:
        print(f"Annotated output saved to: {os.path.join(args.project, args.name)}")
    
if __name__ == '__main__':
	main()