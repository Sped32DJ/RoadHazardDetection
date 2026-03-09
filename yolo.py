#!/usr/bin/env python3
"""Run YOLO model on an input file (image or video) and display/save results.

Usage examples:
  python yolo.py /path/to/image.jpg
  python yolo.py /path/to/video.mp4 --show
  python yolo.py            # will prompt for a file path
"""

import argparse
import os
import sys
from ultralytics import YOLO


def parse_args():
	p = argparse.ArgumentParser(description="Run YOLO on an image or video file")
	p.add_argument('source', nargs='?', help='Path to image/video file. If omitted you will be prompted.')
	p.add_argument('-m', '--model', default='models/best.pt', help='Path to YOLO model file (default: models/best.pt)')
	p.add_argument('--show', action='store_true', help='Show results (opens image/window)')
	return p.parse_args()


def print_detections(results, model):
	names = {}
	try:
		names = model.model.names
	except Exception:
		pass

	for idx, res in enumerate(results):
		print(f"Result {idx}:")
		boxes = getattr(res, 'boxes', None)
		if boxes is None or len(boxes) == 0:
			print('  No detections')
			continue

		# boxes.cls, boxes.conf, boxes.xyxy are tensors
		try:
			xyxy = boxes.xyxy.cpu().numpy()
			conf = boxes.conf.cpu().numpy()
			cls = boxes.cls.cpu().numpy()
		except Exception:
			# Fallback: print the boxes object
			print('  Unable to extract tensor values; raw boxes:', boxes)
			continue

		for i in range(len(conf)):
			label = names.get(int(cls[i]), str(int(cls[i]))) if names else str(int(cls[i]))
			print(f"  {i}: {label} conf={conf[i]:.3f} bbox={xyxy[i]}")


def main():
	args = parse_args()

	source = args.source
	if not source:
		try:
			source = input('Enter path to image/video file: ').strip()
		except EOFError:
			print('No input provided. Exiting.', file=sys.stderr)
			sys.exit(1)

	if not source:
		print('No source provided. Exiting.', file=sys.stderr)
		sys.exit(1)

	if not os.path.exists(source):
		print(f"Source file '{source}' not found.", file=sys.stderr)
		sys.exit(1)

	# Load the model
	try:
		model = YOLO(args.model)
	except Exception as e:
		print(f'Failed to load model "{args.model}": {e}', file=sys.stderr)
		sys.exit(2)

	# Run inference
	try:
		results = model(source)
	except Exception as e:
		print(f'Error during inference: {e}', file=sys.stderr)
		sys.exit(3)

	# Print detections to stdout
	print_detections(results, model)

	# Show or save results
	if args.show:
		for r in results:
			try:
				r.show()
			except Exception as e:
				print(f'Unable to show result: {e}', file=sys.stderr)
	else:
		try:
			results.save()  # saves to runs/detect/predict by default
			print('Saved visualized results to runs/detect/predict')
		except Exception as e:
			print(f'Unable to save results: {e}', file=sys.stderr)


if __name__ == '__main__':
	main()