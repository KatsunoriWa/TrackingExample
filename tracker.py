#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
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

def overlapRange(lim1, lim2):
    start = max(lim1[0],  lim2[0])
    stop = min(lim1[1], lim2[1])

    if start > stop:
        return [None, None]
    else:
        return [start, stop]

def overlapRectArea(rect1, rect2):
    """return overlapped area
    rect1:
    rect2:
    """

    left1, right1 = rect1[0], rect1[0]+rect1[2]
    top1, bottom1 = rect1[1], rect1[1]+rect1[3]


    left2, right2 = rect2[0], rect2[0]+rect2[2]
    top2, bottom2 = rect2[1], rect2[1]+rect2[3]

    [left3, right3] = overlapRange([left1, right1], [left2, right2])
    [top3, bottom3] = overlapRange([top1, bottom1], [top2, bottom2])

    if None in (left3, top3, right3, bottom3):
        return 0.0
    else:
        area = (right3-left3)*(bottom3-top3)
        area >= 0.0
        return area
        
def test_overlapRegion():
    lim = overlapRange([0, 10], [0, 10])
    assert lim == [0, 10]
    lim = overlapRange([0, 10], [0, 20])
    assert lim == [0, 10]
    lim = overlapRange([0, 10], [-10, 20])
    assert lim == [0, 10]


    lim = overlapRange([0, 10], [5, 10])
    assert lim == [5, 10]

    lim = overlapRange([0, 10], [5, 20])
    assert lim == [5, 10]

    lim = overlapRange([-10, 10], [5, 20])
    assert lim == [5, 10]


    lim = overlapRange([5, 10], [5, 20])
    assert lim == [5, 10]


def getIoU(rect1, rect2):
    u"""
    return intersection  over union
"""

    area1 = rect1[2]*rect1[3]
    area2 = rect2[2]*rect2[3]
    intersection = overlapRectArea(rect1, rect2)
    assert intersection >= 0
    union = area1+area2 - intersection
    assert union >= 0

    IoU = intersection/float(union)
    assert IoU >= 0
    return IoU
    

def test_getIoU():
    IoU = getIoU([10, 20, 30, 40], [10, 20, 30, 40])
    print IoU
    assert IoU == 1.0
    
    IoU = getIoU([10, 20, 30, 40], [10, 20, 30, 20])
    print IoU
    assert IoU <=0.5+0.01
    assert 0.5 - 0.01 <= IoU

    IoU = getIoU([10, 20, 30, 40], [10, 25, 30, 40])
    print IoU
    assert IoU < 1.0
    assert 0.0 <=IoU


def creatTracker(tracker_type):
    u"""
    create Tracker
    """

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

    return tracker


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print """usage:tracker [moviefile | uvcID]
        """
        sys.exit()

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
    cascade_path = "haarcascade_frontalface_alt.xml"
    if not os.path.isfile(cascade_path):
        print "be sure to get ready for %s" % os.path.basename(cascade_path)
        sys.exit()

    cascade = cv2.CascadeClassifier(cascade_path)

    try:
        num = int(sys.argv[1])
        video = cv2.VideoCapture(num)
    except:
        video = cv2.VideoCapture(sys.argv[1])
#    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()


    test_overlapRegion()
    test_getIoU()

    rects = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    trackers = range(len(rects))

    for i, rect in enumerate(rects):
        trackers[i] = creatTracker(tracker_type)
        ok = trackers[i].init(frame, tuple(rects[i]))

    while True:
        ok, frame = video.read()
        if not ok:
            break


        rects = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        alreadyFounds = len(rects)*[0.0]

        for i, tracker in enumerate(trackers):
            ok, bbox = tracker.update(frame)
            if ok:            # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            for j, rect in enumerate(rects):
                IoU = getIoU(bbox, rect)
                assert IoU >= 0.0
                assert IoU < 1.0
                alreadyFounds[j] = max(alreadyFounds[j], IoU)

        print rects, alreadyFounds

        for j, alreadyFound in enumerate(alreadyFounds):
            if alreadyFound < 0.5:
                print rects[j]
                x,y,w,h = rects[j]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2, 1)


#        for i, alreadyFound in enumerate(alreadyFounds):
#            if not alreadyFound:
#                tracker = creatTracker(tracker_type)
#                print rects[i]
#                x,y,w,h = rects[i]
#                ok = tracker.init(frame, (long(x), long(y), long(w), long(h)))
#                trackers.append(tracker)

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);


        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break