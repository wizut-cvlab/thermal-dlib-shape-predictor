# Compare dlib shape detector on images in diferent spectrums

In this project we train dlib shape predictor to predict face shapes on thermal images. We want to compare our predictor with common using face shape prediction which was train on normal images (visible spectrum).

For face detection we use simple Haar-like features cascade.

## Runing scripts
Dependencies:
```
pip install numpy
pip install opencv-python
pip install imutils
pip install dlib
```
For dlib installation you may need to install several additional tools, like `cmake` and `Visual Studio build tools`.

Runing:
```
py .\run.py
```

## Images dataset
Inside folder `/images` we assume exist several folders for different models. On each folder may exist various pair images - Thermal and RGB, in the same size. The name for each images consist of 4-letter image identiefier, and suffix indicating image type.

All image datasets is located into `images.7z` archive. The archive is located on: https://cvlab.zut.edu.pl/thermal-dlib-shape-predictor/images.7z

## Predictors
Due to the large size of the files, please download and unzip predictors in application root folder:
https://cvlab.zut.edu.pl/thermal-dlib-shape-predictor/predictors.7z

## Results
The scripts for each image Pari return following summary:
```
#004 #003 [3.5686937077518217, 0.0, 13.0]
```
It means, in folder `#004`, comparing images `#003`, face shape prediction has:
-mean distance `3.5686937077518217`,
-`0.0` minimum distance,
-`13.0` maximum distance,
between two corresponding shape points. Distance is expressed in pixels.

There is 4 results file:
- `results-manual.txt` contains results of manula face marking on Thermal images,
- `results-thermal.txt` contains results of cascade of thermal faces,
- `results-visual.txt` contains results, when for prediction thermal shape we use face cascade generated on visual image.
- `results-pointsDistances.txt` contains calculated distances, for every 68point from all images
- `results-normalizationParams.txt` contains calculated distances used to normalization data

`podsumowanie.xlsx` is additional Excel file with summary results and polish labels.

In `results` folder, there is `manual-boxes.txt` file, where listed manual face marking box coordinates.

## Inside scirpts

### `scripts/draw_shape.py`
Function `draw_shape` gets:
```
img,          - image in grayscales to detect and predict face shape
cascade,      - xml with Haar like feature cascade detector
predictor,    - trained dlib shape predictor
show          - flag, toggle display image in window (default set to false)
manualMarking - flag, allows to Manual marking face (instead using Haar cascade)(default set to false)
box           - Coordinates of detected face (from different visual spectrum or file) (default set to None)
```
Function return `shape` object as numpy array of 68 facial feature points, and `box` array contains coordinates of detected face, or empty array, when there is no face detection.
`normalizationParams` returns 2 values, distances between the outer eye corners (points 36 and 45) and diagonal of the bounding box.

### `scripts/compare_files.py`
For pair of thermal and visual images, function `compare_files` calculates distances (in pixels) between shapes predicted in `draw_shape` function.

### `scripts/face_detect.py`
Legacy script. Allow to generate detection video file.
