# RoadHazardDetection
We have a Jetson Nano running to detect RoadHazards with an emphasis on Potholes

* Make sure you minimun, have an 8Gb swapfile for a smoother experience

## What model are we using?
* We are using the Yolov8 nano model as our base, 
trained in a pothole dataset [link to dataset](https://universe.roboflow.com/brad-dwyer/pothole-voxrl/)
* We trained yolov8n using the above dataset with 100 epochs and downsizing the images to 640
* Offers 55-60ms inference time, giving us 14FPS. Sufficient for real time.

## Setup:
* 

## Usage: `yolo.py`

Follow these steps to run inference with the included `yolo.py` script.

- **Create a virtual environment (recommended):**
 	1. Create a venv using Python 3:
 	```fish
 	python3 -m venv .venv
 	```
 	2. Activate the venv (fish shell):
 	```fish
 	source .venv/bin/activate.fish
 	```
 	(If you use `bash`/`sh` replace the activate command with `source .venv/bin/activate`.)

- **Install dependencies:** Install `ultralytics` and `firebase-admin` (and its requirements) into your Python environment.
 	```fish
 	python -m pip install --upgrade pip
 	python -m pip install ultralytics firebase-admin
	pip install -r requirements-firestore.txt
 	```
- *** For firestore to work, the following instructions are required ***
* 1. Firebase Console > Project Settings > Service Accounts
* 2. Click 'generate new private key'
* 3. Save it as 'firebase-credentials.json' and save it in the same directory as yolo.py.
* Things to note, the pre-requisite to above is having a set up project database with the Firebase API enabled.

- **Default model path:** The script defaults to `models/best.pt`. If you trained or placed your model elsewhere, pass it with `--model`.
 * This is our pre-trained model, trained on the given pothole dataset

- **Training instructions:**
 ``fish
yolo detect train data=Pothole.v1-raw.yolov8/data.yaml epochs=100 imgsz=640
```
* This outputs the new model in runs/detect/train/weights
* Our models/best.pt was trained on this very dataset using the command above

- **Run the script (provide file argument):**
	```fish
	python yolo.py /path/to/image_or_video.mp4
	```

- **Run and show detections interactively:** Use `--show` to open image/video windows (not recommended on headless devices or for video unless you like spam):
	```fish
	python yolo.py /path/to/image_or_video.mp4 --show
	```

- **Prompt for a file path:** If you run without arguments the script will prompt you:
	```fish
	python yolo.py
	# Enter path to image/video file: /path/to/file
	```

- **Saved visualizations:** By default the script saves visualized results to `runs/detect/predict` (same behavior as Ultralytics' `.save()`).

- **Example with custom model and showing results:**
	```fish
	python yolo.py ~/Downloads/video.mp4 --model runs/detect/train/weights/best.pt --show
	```
- **How I typically used it:**

    ```fish
    python yolo.py media/1348.mp4
    ```
    * This outputs into runs/detect/runs/detect/predict/1348.mp4

- *** To detect potholes, send the output to Firestore (timestamp, gps, cropped pothole image, confidence, filename), and show the bounding boxes in the orginal media ***
	```fish
	python yolo.py media/CenoteChichen.jpg --firestore
	```

Notes
- If you are running on a headless Jetson Nano or other server, avoid `--show` and rely on saved outputs.
- If the environment raises "Import 'ultralytics' could not be resolved", install the package into the active Python environment where you run the script.
