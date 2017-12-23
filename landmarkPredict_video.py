#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
#usage :python landmarkPredict.py predictVideo  testList.txt

import os
import sys
import time
import cv2
import dlib # http://dlib.net
import tracker_dlib

import facePose

"""
In this module
bbox = [left, right, top, bottom]
"""

pose_name = ['Pitch', 'Yaw', 'Roll']     # respect to  ['head down','out of plane left','in plane right']

outDir = os.path.expanduser("~/output")
cropDir = os.path.expanduser("~/crop")

def show_image(img, landmarks, bboxs, headposes):
    u"""
    img:
    landmarks: landmark points
    bboxs: dlibの顔検出枠を bounding box としたもののリスト
    headposes:
        headposes[0, :]: 0番目の顔のpitch, yaw, row 
        pitchの値が大きくなると　顎を引いた画像、あるいは上から見下ろした画像になる。
        yawの値が大きくなると　顔向きが画像上の左側を向くようになる。
        rollの値が大きくなると　　顔が時計まわりに傾いた画像になる。 
    """

    orgImg = img+0


    system_height = 650
    system_width = 1280


    for faceNum in range(0, landmarks.shape[0]):
        cv2.rectangle(img, (int(bboxs[faceNum, 0]), int(bboxs[faceNum, 2])), (int(bboxs[faceNum, 1]), int(bboxs[faceNum, 3])), (0, 0, 255), 2)
        for p in range(0, 3):

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, '{:s} {:.2f}'.format(pose_name[p], headposes[faceNum, p]), (10, 400+25*p), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(orgImg, '{:s} {:.2f}'.format(pose_name[p], headposes[faceNum, p]), (10, 400+25*p), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        for i in range(0, landmarks.shape[1]/2):
            cv2.circle(img, (int(round(landmarks[faceNum, i*2])), int(round(landmarks[faceNum, i*2+1]))), 1, (0, 255, 0), 2)
            
        pitch = headposes[faceNum, 0]
        yaw = headposes[faceNum, 1]
        roll = headposes[faceNum, 2]

        pyrStr = facePose.getPyrStr(pitch, yaw, roll)
        pyStr = facePose.getPyStr(pitch, yaw)
        cropPyDir = os.path.join(cropDir, pyStr)
        outPyDir = os.path.join(outDir, pyStr)
        for p in (cropPyDir, outPyDir):
            if not os.path.isdir(p):
                os.makedirs(p)
                
        left, right,  top, bottom = bboxs[faceNum, :]
        rect = [left, top, right-left, bottom - top]
        nx, ny, nw, nh = tracker_dlib.expandRegion(rect, rate=2.0)
        nleft, ntop, nright, nbottom = nx, ny, nx+nw, ny+nh
        assert ntop < nbottom
        assert nleft < nright

        datetimeStr = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        subImg3 = tracker_dlib.sizedCrop(orgImg, (nleft, ntop, nright, nbottom))
        cropName3 = os.path.join(cropPyDir, "%s_%s_b.png" % (pyrStr, datetimeStr))
        cv2.imwrite(cropName3, subImg3)
        

        pngname = os.path.join(outPyDir, "%s_%s.jpg" % (pyrStr, datetimeStr))
        cv2.imwrite(pngname, orgImg)

    if landmarks.shape[0] < 1:
        pyrDir = "couldNotDetect"
        pyrDir = os.path.join(outDir, pyrDir)
        if not os.path.isdir(pyrDir):
            os.makedirs(pyrDir)

        datetimeStr = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        pngname = os.path.join(pyrDir, "%s.jpg" % datetimeStr)
        cv2.imwrite(pngname, orgImg)
        print pngname


    height, width = img.shape[:2]
    if height > system_height or width > system_width:
        height_radius = system_height*1.0/height
        width_radius = system_width*1.0/width
        radius = min(height_radius, width_radius)
        img = cv2.resize(img, (0, 0), fx=radius, fy=radius)

    cv2.imshow("img", img)



def predictVideo(uvcID):
    """
    uvcID: video camera ID
    """

    detector = dlib.get_frontal_face_detector()
    posePredictor = facePose.FacePosePredictor()

    cap = cv2.VideoCapture(uvcID)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    while True:
        ok, colorImage = cap.read()
        if not ok:
            continue

        numUpSampling = 0
        dets, scores, idx = detector.run(colorImage, numUpSampling)
        bboxs = facePose.dets2xxyys(dets)

        predictpoints, landmarks, predictpose = posePredictor.predict(colorImage, bboxs)

        show_image(colorImage, landmarks, bboxs, predictpose)

        k = cv2.waitKey(10) & 0xff
        if k == ord('q') or k == 27:
            break

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print """usage: %s uvcID
        """ % sys.argv[0]
        exit()

    uvcID = int(sys.argv[1])
    predictVideo(uvcID)

