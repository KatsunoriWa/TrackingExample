# TrackingExample
Tracking example code using OpenCV Tracker class in python.

## Requirement
- Python 2.7
- OpenCV3.x or OpenCV4.0.0 pre
- dlib
- caffe

## preparation
### Haar Cascade
Copy following files from OpenCV.

```
haarcascade_frontalface_alt.xml      
haarcascade_frontalface_default.xml
haarcascade_mcs_upperbody.xml
```


### resnet SSD face
Copy following files from OpenCV.

```
face_detector/deploy.prototxt
face_detector/how_to_train_face_detector.txt
face_detector/res10_300x300_ssd_iter_140000.caffemodel
face_detector/solver.prototxt
face_detector/test.prototxt
face_detector/train.prototxt
```

### dlib frontal face
Copy following files from dlib.

shape_predictor_68_face_landmarks.dat


## usage:
libtracker.py
```
$ python libtracker.py
usage:libtracker.py [moviefile | uvcID]
```
cv2.__version__ 4.0.0-pre
$

### 検出エンジンの切り替え
スクリプト中のdetectorTypeの値を変更することで、検出エンジンを切り替えることができる。
- Haar Cascade
- resnet SSD face detection version
- dlib frontal face

### 追跡手法の切り替え
スクリプト中のtracker_typeの値を切り替えることで、追跡手法を切り替えることができる。
'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN'




### advanced face processing version based on dlib face detection


```
$ ./tracker_dlib_pose.py
usage:./tracker_dlib_pose.py  [--crop] (moviefile | uvcID)
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
