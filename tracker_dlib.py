#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# Python 2/3 compatibility
from __future__ import print_function
import sys
import os
import time
import numpy as np
import cv2
import dlib

import librect
import libtracker
import facePose

def draw_landmarks(frame, shape):
    """
    frame: alignedImg
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

if __name__ == '__main__':
    import getopt
    optlist, sys.argv[1:] = getopt.getopt(sys.argv[1:], '', ['crop', 'align', 'saveFull'])
    if len(sys.argv) == 1:
        print("""usage:%s  [--crop] (moviefile | uvcID)
--crop: enable crop
--align: enable aligne
--saveFull: save full image
        """ % sys.argv[0])
        print("cv2.__version__", cv2.__version__)
        sys.exit()


    try:
        num = int(sys.argv[1])
        video = cv2.VideoCapture(num)
    except:
        video = cv2.VideoCapture(sys.argv[1])

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    doCrop = False
    if ('--crop', '') in optlist:
        doCrop = True

    doAlign = False
    if ('--align', '') in optlist:
        doAlign = True


    doSaveFull = False
    if ('--saveFull', '') in optlist:
        doSaveFull = True


    cropDir = "crop"
    if not os.path.isdir(cropDir):
        os.makedirs(cropDir)

    fullImgDir = "fullImg"
    if not os.path.isdir(fullImgDir):
        os.makedirs(fullImgDir)

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    librect.test_overlapRegion()
    librect.test_getIoU()

    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    #<dlib>
    detector = dlib.get_frontal_face_detector()
    numUpSampling = 0

    posePredictor = facePose.FacePosePredictor()

    dets, scores, idx = detector.run(frame, numUpSampling)
    rects = librect.dets2rects(dets)
    #</dlib>

    print(rects)

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    trackers = range(len(rects))

    for i, rect in enumerate(rects):
        trackers[i] = libtracker.TrackerWithState(tracker_type)
        ok = trackers[i].init(frame, tuple(rects[i]))

    counter = 0

    interval = 4

    color = {True:(0, 0, 255), False:(255, 0, 0)}
    while True:
        ok, frame = video.read()
        if not ok:
            break

        frameCopy = frame+0

        doDetect = (counter % interval == interval - 1)

        indexes = range(len(trackers))
        indexes.reverse()
        for i in indexes:
            tracker = trackers[i]
            #  追跡する。
            trackOk, bbox = tracker.update(frame)
            if trackOk:            # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, color[doDetect], 2, 1)

                left, top, w, h = bbox
                right, bottom = left+w, top+h
                det = dlib.rectangle(long(left), long(top), long(right), long(bottom))
                shape = predictor(frame, det)
                frame = draw_landmarks(frame, shape)
                stateStr = {True:"detect  frame", False:"no detect  frame"}
                cv2.putText(frame, stateStr[doDetect], (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color[doDetect], 2)
            else:
                del trackers[i]
                print("""del trackers["%d"] """ % i)
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                continue

            if doCrop:
                predictpoints, landmarks, headposes = posePredictor.predict(frameCopy, np.array([[left, right, top, bottom]]))

                pitch = headposes[0, 0]
                yaw = headposes[0, 1]
                roll = headposes[0, 2]

                cropPyrDir = facePose.getPyrDir(cropDir, pitch, yaw, roll)

                print(pitch, yaw, roll, "# pitch, yaw, roll")
                print(cropPyrDir)

                subImg = frameCopy[int(top):int(bottom), int(left):int(right), :]

                datetimestring = time.strftime("%Y%m%d_%H%M%S", time.localtime())

                nx, ny, nw, nh = librect.expandRegion(rect, rate=2.0)
                nleft, ntop, nright, nbottom = nx, ny, nx+nw, ny+nh
                assert ntop < nbottom
                assert nleft < nright

                subImg3 = librect.sizedCrop(frameCopy, (nleft, ntop, nright, nbottom))
                cropName3 = os.path.join(cropPyrDir, "%s_b.png" % datetimestring)
                cv2.imwrite(cropName3, subImg3)
                print(cropName3)

            if doAlign:
                alignedImg = dlib.get_face_chip(frameCopy, shape, size=320, padding=0.5)
                alignedImg = np.array(alignedImg, dtype=np.uint8)
                cv2.imshow('alignedImg', alignedImg)
                cropName4 = os.path.join(cropPyrDir, "%s_aligned.png" % datetimestring)
                cv2.imwrite(cropName4, alignedImg)
                print(cropName4)
                k = cv2.waitKey(1) & 0xff
                if k == ord('q') or k == 27:
                    break

        if doDetect:
            #<dlib>
            dets, scores, idx = detector.run(frame, numUpSampling)
            faces = dlib.full_object_detections()
            for detection in dets:
                    faces.append(predictor(frame, detection))

            for face in faces:
                alignedImg = dlib.get_face_chip(frameCopy, face, size=320, padding=0.5)
                alignedImg = np.array(alignedImg, dtype=np.uint8)
                cv2.imshow('alignedImg', alignedImg)
                k = cv2.waitKey(1) & 0xff
                if k == ord('q') or k == 27:
                    break

            rects = librect.dets2rects(dets)
            print(rects, scores, idx)
            #</dlib>

            # どれかの検出に重なっているかを調べる。
            # 一番重なりがよいのを見つける。
            # 一番重なりがよいものが、しきい値以上のＩｏＵだったら、追跡の位置を初期化する。
            # 一番の重なりのよいものが一定値未満だったら、新規の追跡を開始する。
            states = [(t.ok, t.bbox) for t in trackers]
            alreadyFounds, asTrack = librect.getBestIoU(rects, states)

            for j, rect in enumerate(rects):# 検出について
                if alreadyFounds[j] > 0.5:
                    # 十分に重なっていて検出結果で追跡を置き換える。
                    print(librect.rect2bbox(rect), "# rect2bbox(rect)")
                    ok = trackers[asTrack[j]].init(frame, librect.rect2bbox(rect))
                    left, top, w, h = rect
                    right, bottom = left+w, top+h
                    det = dlib.rectangle(left, top, right, bottom)
                    shape = predictor(frame, det)
                    frame = draw_landmarks(frame, shape)
                    [left, right, top, bottom] = [det.left(), det.right(), det.top(), det.bottom()]
                elif alreadyFounds[j] < 0.5 - 0.1:
                    # 対応する追跡がないとして、新規の検出にする。
                    tracker = libtracker.TrackerWithState(tracker_type)
                    ok = tracker.init(frame, librect.rect2bbox(rects[j]))
                    trackers.append(tracker)
                    print("new tracking")
                    print(librect.rect2bbox(rect), "# rect2bbox(rect) new tracking")
                    left, top, w, h = rects[j]
                    right, bottom = left+w, top+h
                    det = dlib.rectangle(left, top, right, bottom)
                    shape = predictor(frame, det)
                else:
                    [left, right, top, bottom] = [det.left(), det.right(), det.top(), det.bottom()]


        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        cv2.putText(frame, "# of Trackers = %d" % len(trackers), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);


        cv2.namedWindow("Tracking q:quit", cv2.WINDOW_NORMAL)
        cv2.imshow("Tracking q:quit", frame)
        counter += 1
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cv2.destroyAllWindows()
    video.release()