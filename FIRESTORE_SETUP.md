# Google Firestore Integration Setup Guide

This guide walks you through setting up Google Cloud Firestore integration with your YOLO pothole detection script.

## Prerequisites

### 1. Install Required Python Packages

```bash
pip install firebase-admin pillow piexif
```

These packages provide:
- **firebase-admin**: Google Cloud Firestore integration
- **pillow**: Image manipulation (cropping, resizing)
- **piexif**: EXIF metadata extraction (for GPS data)

### 2. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable **Cloud Firestore API**:
   - Click **Enable APIs and Services**
   - Search for "Cloud Firestore"
   - Click **Enable**

### 3. Create a Service Account and Download Credentials

1. In Google Cloud Console, go to **Service Accounts**:
   - Menu → IAM & Admin → Service Accounts
   
2. Click **Create Service Account**
   - Service Account Name: `yolo-detection`
   - Click **Create and continue**

3. Grant permissions:
   - Role: Select **Cloud Datastore User** (or **Editor** for full access)
   - Click **Continue** → **Done**

4. Generate a private key:
   - Click your new service account
   - Go to **Keys** tab
   - Click **Add Key** → **Create new key**
   - Select **JSON**
   - A JSON file will download automatically
   
5. Save the JSON file:
   ```bash
   # Move the downloaded JSON file to your project directory
   mv ~/Downloads/[service-account-name].json ~/projects/RoadHazardDetection/firebase-credentials.json
   ```

### 4. Create a Firestore Database

1. In Google Cloud Console, go to **Firestore**
2. Click **Create Database**
3. Start in **Production mode**
4. Choose a location (e.g., `us-central1`)
5. Click **Create Database**

## Usage

### Basic Usage - Send Detections to Firestore

```bash
# Run detection and upload to Firestore
python yolo.py /path/to/image.jpg --firestore

# With custom model
python yolo.py /path/to/image.jpg --firestore -m models/best.pt

# With cropped pothole images
python yolo.py /path/to/image.jpg --firestore --crop-images

# Custom collection name
python yolo.py /path/to/image.jpg --firestore --collection pothole_reports

# Custom credentials file location
python yolo.py /path/to/image.jpg --firestore --creds /path/to/creds.json
```

### Firestore Data Structure

Each detection creates a document in Firestore with the following structure:

```
Collection: "detections"
Document: {
  timestamp: 2026-03-11T15:30:45.123Z,
  source_file: "/path/to/image.jpg",
  class_label: "pothole",
  class_id: 0,
  confidence: 0.92,
  bbox: {
    x1: 100.5,
    y1: 200.3,
    x2: 250.8,
    y2: 350.2
  },
  gps: {
    latitude: 40.7128,
    longitude: -74.0060,
    altitude: 10.5
  },
  cropped_image_base64: "iVBORw0KGgoAAAANSUhEUgAAAAUA...",
  image_size: "150x150"
}
```

**Field Explanations:**
- `timestamp`: When the detection was made
- `source_file`: Path to the source image/video
- `class_label`: Class name (e.g., "pothole")
- `class_id`: Numeric class ID
- `confidence`: Detection confidence score (0-1)
- `bbox`: Bounding box coordinates [x1, y1, x2, y2]
- `gps`: GPS coordinates extracted from image EXIF (if available)
- `cropped_image_base64`: Base64-encoded cropped image (if `--crop-images` used)
- `image_size`: Dimensions of the cropped image

## GPS Data Extraction

The script automatically extracts GPS coordinates from image EXIF metadata if:
1. The source file is a JPEG or PNG image
2. The image contains GPS EXIF data
3. The `piexif` library is installed

**Note:** Most smartphone cameras embed GPS data. Dashcam videos may also have GPS in frames.

## Uploading Cropped Images

With `--crop-images`, the script:
1. Crops the detected object with padding
2. Compresses it to max 400x400 pixels
3. Converts to JPEG (quality 70) to save space
4. Encodes as base64 for Firestore storage

**Storage Considerations:**
- Firestore has a 1MB document size limit
- Base64 images are ~25-30% larger than binary
- A 400x400 JPEG is typically 20-50KB when base64 encoded
- Each detection document is well under the limit

To retrieve and decode images later:

```python
import base64
from PIL import Image
from io import BytesIO

# From Firestore document
img_base64 = doc['cropped_image_base64']
img_data = base64.b64decode(img_base64)
image = Image.open(BytesIO(img_data))
image.save('pothole.jpg')
```

## Querying Firestore

### Example: Get all detections above 0.9 confidence

```python
from firebase_admin import credentials, firestore, initialize_app

cred = credentials.Certificate('firebase-credentials.json')
initialize_app(cred)
db = firestore.client()

results = db.collection('detections').where('confidence', '>=', 0.9).stream()
for doc in results:
    print(doc.to_dict())
```

### Example: Get detections with GPS data

```python
results = db.collection('detections').where('gps', '!=', None).stream()
for doc in results:
    data = doc.to_dict()
    gps = data['gps']
    print(f"Pothole at ({gps['latitude']}, {gps['longitude']})")
```

### Example: Get recent detections

```python
from datetime import datetime, timedelta

yesterday = datetime.now() - timedelta(days=1)
results = db.collection('detections').where(
    'timestamp', '>=', yesterday
).stream()

for doc in results:
    print(doc.to_dict())
```

## Troubleshooting

### "firebase-admin not installed"
```bash
pip install firebase-admin
```

### "Firebase credentials file not found"
Make sure `firebase-credentials.json` is in your project directory, or specify with:
```bash
python yolo.py image.jpg --firestore --creds /path/to/creds.json
```

### "Could not extract GPS data"
- Image doesn't contain GPS EXIF data
- Image format doesn't support EXIF (some PNGs)
- This is logged as a warning and won't stop processing

### Firestore quota exceeded
- Check your Firestore usage in Google Cloud Console
- Free tier allows 50k reads/day (usually enough for testing)
- Documents are small (~1KB without images)

### Document too large
- The 1MB limit is rarely hit even with base64 images
- If it happens, decrease image quality or don't use `--crop-images`

## Security Considerations

1. **Keep credentials secret**: Never commit `firebase-credentials.json` to git
   ```bash
   # Add to .gitignore
   echo "firebase-credentials.json" >> .gitignore
   ```

2. **Limit service account permissions**: Give it only Firestore write access

3. **Use Firestore security rules** to restrict who can read/write data:
   ```
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       // Only allow writes from authorized service accounts
       match /detections/{document=**} {
         allow read, write: if request.auth != null;
       }
     }
   }
   ```

## Cost Estimation

Firestore pricing (as of 2024):
- **Reads**: $0.06 per 100k documents
- **Writes**: $0.18 per 100k documents
- **Delete**: $0.02 per 100k documents
- **Storage**: $0.18 per GB/month

For road monitoring with 100 daily detections:
- ~3,000 detections/month
- ~$0.00054 in write costs (negligible)
- Storage: ~3MB = minimal

## Advanced Usage

### Custom Collection Names
```bash
python yolo.py image.jpg --firestore --collection my_potholes
```

### Video Processing
The script can process videos:
```bash
python yolo.py /path/to/video.mp4 --firestore --crop-images
```
Each detected frame creates a document with detections.

### Batch Processing
```bash
for file in *.jpg; do
    python yolo.py "$file" --firestore --crop-images
done
```

## Next Steps

1. Test with a single image:
   ```bash
   python yolo.py test_image.jpg --firestore
   ```

2. Check Firestore Console to verify data appears

3. Build a dashboard or web app to visualize detections

4. Set up alerts for high-confidence detections

## Resources

- [Firestore Documentation](https://firebase.google.com/docs/firestore)
- [Firebase Admin SDK Python](https://firebase.google.com/docs/database/admin/start)
- [EXIF Data Extraction](https://piexif.readthedocs.io/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
