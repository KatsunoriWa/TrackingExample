# TrackingExample
Tracking example code using OpenCV Tracker class in python.

## Requirement
- Python 2.7
- OpenCV3.x
- dlib

## usage:

dlib face detection version

```
$ ./tracker_dlib.py
usage:./tracker_dlib.py  [--crop] (moviefile | uvcID)
--crop: enable crop
--align: enable aligne
--saveFull: save full image

$
$ python tracker_dlib.py 0
$
```


OpenCV face detection version

```
$ python tracker.py
usage:tracker.py [moviefile | uvcID]

$
$ python tracker.py 0
$
```

## Note
 This code is not useful because face detection is not fast enough.
