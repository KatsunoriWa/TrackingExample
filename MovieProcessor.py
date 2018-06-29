# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import os
import time
import cv2 as cv

class MovieProcessor(object):

    def __init__(self, src):
        """
        src: movie file name or cameraID
        """

        def timeStr():
            return time.strftime("%Y%m%d_%H%M%S", time.localtime())

        self.src = src
        if os.path.isfile(src):
            self.base = os.path.splitext(os.path.basename(src))[0]
            self.cap = cv.VideoCapture(self.src)
        else:
            self.base = "%s" % timeStr()
            self.cap = cv.VideoCapture(int(self.src))

    def setOutdir(self, outDir=""):
        if not outDir:
            self.outname = ""
            return

        if not os.path.isdir(outDir):
            os.mkdir(outDir)

        self.outname = os.path.join(outDir, "%s_out.avi" % self.base)
        self.detectionLogName = os.path.join(outDir, "%s_detect.txt" % self.base)

    def processFrame(self, frame):
        return frame

    def process(self, confThreshold):
        """process by detecor
        """

        outname = self.outname
        cap = self.cap
        
        self.detectionLog = open(self.detectionLogName, "wt")        
        
        rec = None
        counter = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            counter += 1

            cols = frame.shape[1]
            rows = frame.shape[0]

            if outname:
                if rec is None:
                    FRAME_RATE = 30
                    rec = cv.VideoWriter(outname, \
                                  cv.VideoWriter_fourcc(*'MJPG'), \
                                  FRAME_RATE, \
                                  (cols, rows))

            frame = self.processFrame(frame, counter)

            cv.imshow("detections", frame)
            if not rec is None:
                rec.write(frame)

            k = cv.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break


        if not rec is None:
            rec.release()
        cap.release()
        
        self.detectionLog.close()