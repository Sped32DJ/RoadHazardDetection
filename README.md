# RoadHazardDetection
We have a Jetson Nano running to detect RoadHazards with an emphasis on Potholes

## What model are we using?
* We are using the Yolov8 nano model as our base, 
trained in a pothole dataset [link to dataset](https://universe.roboflow.com/brad-dwyer/pothole-voxrl/)
* We trained yolov8n using the above dataset with 100 epochs and downsizing the images to 640
* Offers 70ms inference time, giving us 14FPS. Sufficient for real time.

## Usage: `yolo.py`

Follow these steps to run inference with the included `yolo.py` script.

- **Install dependencies:** Install `ultralytics` (and its requirements) into your Python environment.
	```fish
	python -m pip install --upgrade pip
	python -m pip install ultralytics
	```

- **Default model path:** The script defaults to `models/best.pt`. If you trained or placed your model elsewhere, pass it with `--model`.

- **Training instructions:**
 ``fish
yolo detect train data=Pothole.v1-raw.yolov8/data.yaml epochs=100 imgsz=640
```
* This outputs the new model in runs/detect/train/weights

- **Run the script (provide file argument):**
	```fish
	python yolo.py /path/to/image_or_video.mp4
	```

- **Run and show detections interactively:** Use `--show` to open image/video windows (not recommended on headless devices):
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

Notes
- If you are running on a headless Jetson Nano or other server, avoid `--show` and rely on saved outputs.
- If the environment raises "Import 'ultralytics' could not be resolved", install the package into the active Python environment where you run the script.
