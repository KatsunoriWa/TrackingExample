#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
import sys
import os
import cv2
import dlib

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
    """return overlapped lim
    lim1:
    lim2:
    """

    start = max(lim1[0], lim2[0])
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
    assert IoU <= 0.5+0.01
    assert 0.5 - 0.01 <= IoU

    IoU = getIoU([10, 20, 30, 40], [10, 25, 30, 40])
    print IoU
    assert IoU < 1.0
    assert IoU >= 0.0


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

def rect2bbox(rect):
    """convert rect into bbox.
    tracker.init() need this data type.
    """

    assert len(rect) == 4
    x, y, w, h = rect
    assert w > 0
    assert h > 0
    return (long(x), long(y), long(w), long(h))


def dets2rects(dets):
    """
    convert dets type to rect type.
"""

    rects = [[d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()] for d in dets]
    return rects


def getBestIoU(rects, states):
    u"""find best matched tracking for each rect.
    rects: detected rects
    states: tracking states
    
    """
    
    asTrack = len(rects)*[None]
    alreadyFounds = len(rects)*[0.0]

    for j, rect in enumerate(rects):# 検出について
        for k, (_, bbox) in  enumerate(states):#追跡について
            IoU = getIoU(bbox, rect)
            assert IoU >= 0.0
            assert len(rect) == 4
            assert rect[2] > 0
            assert rect[3] > 0
            if IoU > alreadyFounds[j]:
                alreadyFounds[j] = max(alreadyFounds[j], IoU)
                asTrack[j] = k
    return alreadyFounds, asTrack

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print """usage:tracker [moviefile | uvcID]
        """
        sys.exit()

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

    try:
        num = int(sys.argv[1])
        video = cv2.VideoCapture(num)
    except:
        video = cv2.VideoCapture(sys.argv[1])

    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()

    test_overlapRegion()
    test_getIoU()

    detector = dlib.get_frontal_face_detector()
    rects = dets2rects(detector(frame, 1))

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    trackers = range(len(rects))

    for i, rect in enumerate(rects):
        trackers[i] = creatTracker(tracker_type)
        ok = trackers[i].init(frame, tuple(rects[i]))

    counter = 0

    interval = 20


    color = {True:(0, 0, 255), False:(255, 0, 0)}
    while True:
        ok, frame = video.read()
        if not ok:
            break

        doDetect = (counter % interval == interval - 1)

        states = []
        for i, tracker in enumerate(trackers):
            #  追跡する。
            trackOk, bbox = tracker.update(frame)
            states.append((trackOk, bbox))
            if ok:            # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, color[doDetect], 2, 1)
                if doDetect:
                    cv2.putText(frame, "detect  frame", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color[doDetect], 2)
                else:
                    cv2.putText(frame, "no detect  frame", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color[doDetect], 2)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


        if doDetect:
            rects = dets2rects(detector(frame, 1))

            # どれかの検出に重なっているかを調べる。
            # 一番重なりがよいのを見つける。
            # 一番重なりがよいものが、しきい値以上のＩｏＵだったら、追跡の位置を初期化する。
            # 一番の重なりのよいものが一定値未満だったら、新規の追跡を開始する。

            alreadyFounds, asTrack = getBestIoU(rects, states)

            for j, rect in enumerate(rects):# 検出について
                if alreadyFounds[j] > 0.5:
                    print rect2bbox(rect), "# rect2bbox(rect)"
                    ok = trackers[asTrack[j]].init(frame, rect2bbox(rect))
                elif alreadyFounds[j] < 0.5 - 0.1:
                    tracker = creatTracker(tracker_type)
                    ok = tracker.init(frame, rect2bbox(rects[j]))
                    trackers.append(tracker)
                    print "new tracking"
                else:
                    continue

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        cv2.putText(frame, "# of Trackers = %d" % len(trackers), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);


        cv2.imshow("Tracking q:quit", frame)
        counter += 1
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break