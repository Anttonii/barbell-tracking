## Barbell Path Tracking

An experimental project to track the path of a barbell from a video file and then translating the path to useful data such as:
 - a plot that represents the speed of the bar over time.
 - the deviation from the mean or how far from straight path down the bar travels and in which direction.

Refer to `TODO.md` to get an idea of what is being done and what is left to be done.

## Technicalities

This project uses YOLOv3 to first detect a barbell then DeepSort to track the object over the video frames.

### YOLOv3 demo
| Test Image | Detection Result |
| --- | --- |
| ![Test image](/dog-cycle-car.png) | ![Detection image](/output/det_dog-cycle-car.png) |

## Installation

Install all the requirements first:

```
pip install -r requirements.txt
```

then make sure you have the `yolov3.weights` in the data folder, which can be acquired from here:

```
wget https://pjreddie.com/media/files/yolov3.weights
```

All other files are supplied as a part of this repository.

## Usage

Currently the project only supports YOLOv3 object detection, which can be tested with:

```
python tracker.py
```

This command takes images in `imgs` folder and makes runs the YOLOv3 algorithm outputting images with bounding boxes showing the detections made into the `output` folder. You can also run the program for a custom paths for input and output using:

```
python tracker.py detect --images <your_image_paths_here> --output output
```

For the full list of options that can be configured run the following command:

```
python tracker.py --help
```

**Note:** when changing the resolution option, make sure that the given resolution is divisible by 32.

## Licensing

This project is under MIT licensing.

## Credits

Thanks to
 - Ayoosh Kathuria for their [5-part series](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/) on YOLOv3 implementation.
 - Kapil Sachdeva for their [video series](https://www.youtube.com/watch?v=6LOsCEs9IAc) on object detection.