#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# Python 2/3 compatibility
from __future__ import print_function

import sys
import os
import time
import cv2
import dlib
import librect # GX
from resnetFaceDetector import ResnetFaceDetector # GX


class TrackerWithState(object):
    """
    tracker
    """

    def __init__(self, tracker_type):
        """create tracter instance
        tracker_type:
        """

        self.cvTracker = self._creatTracker(tracker_type)
        self.ok = False
        self.bbox = []

    def update(self, frame):
        """
        frame: frame image
        """

        trackOk, bbox = self.cvTracker.update(frame)
        self.ok = trackOk
        self.bbox = bbox
        return trackOk, bbox

    def init(self, frame, rect):
        """
        initialize tracker instance with frame and rect
        frame:
        rect:
        """

        self.cvTracker.init(frame, rect)
        self.ok = True
        self.bbox = rect


    def _creatTracker(self, tracker_type):
        u"""
        create Tracker
        """
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

        if major_ver == 3 and int(minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()

        return tracker



def draw_landmarks(frame, shape):
    """
    frame: image
    shape: landmark points by dlib.shape_predictor(predictor_path)
    """
    for shape_point_count in range(shape.num_parts):
        shape_point = shape.part(shape_point_count)
        if shape_point_count < 17: # [0-16]:輪郭
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (0, 0, 255), -1)
        elif shape_point_count < 22: # [17-21]眉（右）
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (0, 255, 0), -1)
        elif shape_point_count < 27: # [22-26]眉（左）
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (255, 0, 0), -1)
        elif shape_point_count < 31: # [27-30]鼻背
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (0, 255, 255), -1)
        elif shape_point_count < 36: # [31-35]鼻翼、鼻尖
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (255, 255, 0), -1)
        elif shape_point_count < 42: # [36-4142目47）
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (255, 0, 255), -1)
        elif shape_point_count < 48: # [42-47]目（左）
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (0, 0, 128), -1)
        elif shape_point_count < 55: # [48-54]上唇（上側輪郭）
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (0, 128, 0), -1)
        elif shape_point_count < 60: # [54-59]下唇（下側輪郭）
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (128, 0, 0), -1)
        elif shape_point_count < 65: # [60-64]上唇（下側輪郭）
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (0, 128, 255), -1)
        elif shape_point_count < 68: # [65-67]下唇（上側輪郭）
            cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 2, (128, 255, 0), -1)
    return frame

class HaarCascadeDetector(object):
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml', scaleFactor=1.1, minNeightbors=5):
    	 if not os.path.isfile(cascade_path):
    	     print("be sure to get ready for %s" % os.path.basename(cascade_path))
         sys.exit()

         self.face_cascade = cv2.CascadeClassifier(cascade_path)
         self.scaleFactor = scaleFactor
         self.minNeightbors = minNeightbors

    def run(self, frame):
         dets = self.face_cascade.detectMultiScale(frame, self.scaleFactor, self.minNeightbors)
         return dets, "noScore", "noIdx"

class DlibFrontalDetector(object):
    def __init__(self, numUpSampling=1):
        self.detector = dlib.get_frontal_face_detector()
        self.numUpSampling = numUpSampling

    def run(self, frame):
        dets = self.detector(frame, self.numUpSampling)
        rects = librect.dets2rects(dets)
        return rects, None, None




def getCapAndbase(src):
    """
    src: movie file name or cameraID
    """

    def timeStr():
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if os.path.isfile(src):
        base = os.path.splitext(os.path.basename(src))[0]
        cap = cv2.VideoCapture(src)
    else:
        base = "%s" % timeStr()
        cap = cv2.VideoCapture(int(src))

    return cap, base

def getOutname(base, outDir=""):
    if not outDir:
        outname = ""
        return outname

    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    outname = os.path.join(outDir, "%s_out.avi" % base)
    return outname

def main(src):

    video, base = getCapAndbase(src)

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    detectorType = "resnetSSD"

    if detectorType == "Haar":
        detector = HaarCascadeDetector()
    elif detectorType == "dlib":
        detector = DlibFrontalDetector()
    elif detectorType == "resnetSSD":
        confThreshold = 0.5
        detector = ResnetFaceDetector(confThreshold)
    else:
        print("no such detector")
        exit()


    aviname = getOutname(base, outDir="../track_result")
    vout = None


    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[4]

    trackers = []
    counter = 0

    interval = 20

    color = {True:(0, 0, 255), False:(255, 0, 0)}
    while True:
        ok, frame = video.read()
        if not ok:
            break

        cols = frame.shape[1]
        rows = frame.shape[0]

        if aviname:
            if vout is None:
                FRAME_RATE = 30
                vout = cv2.VideoWriter(aviname, \
                              cv2.VideoWriter_fourcc(*'MJPG'), \
                              FRAME_RATE, \
                              (cols, rows))


        doDetect = (counter % interval == 0)

        indexes = range(len(trackers))
        indexes.reverse()

        usedTracker = 0
        for i in indexes:
            #  追跡する。
            t0 = cv2.getTickCount()
            trackOk, bbox = trackers[i].update(frame)
            t1 = cv2.getTickCount()
            usedTracker += (t1-t0)/cv2.getTickFrequency()
            if trackOk:            # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, color[doDetect], 2, 1)

                left, top, w, h = bbox
                right, bottom = left+w, top+h

            else:
                del trackers[i]
                print("""del trackers["%d"] """ % i)

        if doDetect:
            usedDetector = 0
            t0 = cv2.getTickCount()

            #<haar>
#            rects = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            rects, _, _ = detector.run(frame)

            #</haar>
            t1 = cv2.getTickCount()
            usedDetector += (t1-t0)/cv2.getTickFrequency()

            # どれかの検出に重なっているかを調べる。
            # 一番重なりがよいのを見つける。
            # 一番重なりがよいものが、しきい値以上のＩｏＵだったら、追跡の位置を初期化する。
            # 一番の重なりのよいものが一定値未満だったら、新規の追跡を開始する。
            states = [(t.ok, t.bbox) for t in trackers]
            alreadyFounds, asTrack = librect.getBestIoU(rects, states)

            for j, rect in enumerate(rects):# 検出について
                if alreadyFounds[j] > 0.5:
                    print(librect.rect2bbox(rect), "# rect2bbox(rect)")
                    ok = trackers[asTrack[j]].init(frame, librect.rect2bbox(rect))
                    left, top, w, h = rect
                    right, bottom = left+w, top+h
                elif alreadyFounds[j] < 0.5 - 0.1:
                    tracker = TrackerWithState(tracker_type)
                    ok = tracker.init(frame, librect.rect2bbox(rects[j]))
                    trackers.append(tracker)
                    print("new tracking")
                    print(librect.rect2bbox(rect), "# rect2bbox(rect) new tracking")
                    left, top, w, h = rects[j]
                else:
                    continue

        if doDetect:
            cv2.putText(frame, "   detect frame: used %.3f"  % usedDetector, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color[doDetect], 2)
        else:
            cv2.putText(frame, "no detect frame: used %.3f" % usedTracker, (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color[doDetect], 2)

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        cv2.putText(frame, "# of Trackers = %d" % len(trackers), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        cv2.namedWindow("Tracking q:quit", cv2.WINDOW_NORMAL)
        cv2.imshow("Tracking q:quit", frame)
        vout.write(frame)
        counter += 1
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cv2.destroyAllWindows()
    video.release()
    vout.release()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("""usage:%s [moviefile | uvcID]
        """ % sys.argv[0])
        print("cv2.__version__", cv2.__version__)
        sys.exit()

    main(sys.argv[1])