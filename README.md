# RoadHazardDetection
We have a Jetson Nano running to detect RoadHazards with an emphasis on Potholes

## What model are we using?
* We are using the Yolov8 nano model as our base, 
trained in a pothole dataset [link to dataset](https://universe.roboflow.com/brad-dwyer/pothole-voxrl/)
* We trained yolov8n using the above dataset with 100 epochs and downsizing the images to 640
* Offers 70ms inference time, giving us 14FPS. Sufficient for real time.
