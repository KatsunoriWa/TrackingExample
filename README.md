# TrackingExample
Tracking example code using OpenCV Tracker class in python.

## Requirement
- Python 2.7
- OpenCV3.x
- dlib
- caffe

## usage:

### dlib face detection version

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

以下の処理を利用することで、顔の特徴点を表示させることができる。


evaluate_48.py
wase copied from
https://github.com/RiweiChen/DeepFace/blob/master/FaceDetection/baseline/evaluate_48.py




### OpenCV face detection version

```
$ python tracker.py
usage:tracker.py [moviefile | uvcID]

$
$ python tracker.py 0
$
```

## NoTracking script

landmarkPredict_video.py
Detect faces using  dlib.get_frontal_face_detector(), and get facePose using facePose.py


### modules
facePose.py: a module to predict face pose.


## Note
 This code is not useful because face detection is not fast enough.
