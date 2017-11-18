#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
"""
import sys
import os
import cv2

def largestRect(rects):
    u"""retturn largest rect in rects
    rects: list of rect
    """

    if len(rects) < 2:
        return rects

    largest = rects[0]
    for i in range(1, len(rects)):
        if rects[i][2] > largest[2]:
            largest = rects[i]

    return largest

if __name__ == '__main__':
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
    cascade_path = "haarcascade_frontalface_alt.xml"
    if not os.path.isfile(cascade_path):
        print "be sure to get ready for %s" % os.path.basename(cascade_path)
        sys.exit()

    cascade = cv2.CascadeClassifier(cascade_path)

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
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

#    video = cv2.VideoCapture("videos/chaplin.mp4")
    video = cv2.VideoCapture(0)


    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()

    rects = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    largest = largestRect(rects)
    print largest

    bbox = tuple(largest)

    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()

        ok, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if ok:            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break