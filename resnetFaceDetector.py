#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101


# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
from cv2 import dnn

import MovieProcessor

class ResnetFaceDetector(object):
    """
    example:
    detector = ResnetFaceDetector()
    confThreshold = 0.5
    dets, confidences, perf_stats = self.detector.run(frame, confThreshold)
    """

    def __init__(self, confThreshold=0.5):
        """ResnetFaceDetecor class rewrite version
        """

        prototxt = 'face_detector/deploy.prototxt'
        caffemodel = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
        self.inWidth = 300
        self.inHeight = 300
        self.net = dnn.readNetFromCaffe(prototxt, caffemodel)

        self.confThreshold = confThreshold

    def _run(self, image):
        """
        image: input image
        return: detections, perf_stats
        """

        self.net.setInput(dnn.blobFromImage(image, 1.0, (self.inWidth, self.inHeight), (104.0, 177.0, 123.0), False, False))
        self.detections = self.net.forward()

        self.perf_stats = self.net.getPerfProfile()
        return self.detections, self.perf_stats

    def run(self, image):
        """
        image: input image
        confThreshold:
        return: detections, confidences, perf_stats
        """
        self._run(image)

        cols = image.shape[1]
        rows = image.shape[0]

        detections = self.detections
        confThreshold = self.confThreshold

        dets = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                xLeftTop = int(detections[0, 0, i, 3] * cols)
                yLeftTop = int(detections[0, 0, i, 4] * rows)
                xRightBottom = int(detections[0, 0, i, 5] * cols)
                yRightBottom = int(detections[0, 0, i, 6] * rows)
                dets.append((xLeftTop, yLeftTop, xRightBottom - xLeftTop, yRightBottom - yLeftTop))
                confidences.append(confidence)

        return dets, confidences, self.perf_stats


class ResnetFaceMovieProcessor(MovieProcessor.MovieProcessor):

    def processFrame(self, frame, counter):
        dets, confidences, perf_stats = self.detector.run(frame)
        print('Inference time, ms: %.2f' % (perf_stats[0] / cv.getTickFrequency() * 1000))

        for i, det in enumerate(dets):
            confidence = confidences[i]
            xLeftTop, yLeftTop, w, h = det
            xRightBottom = xLeftTop + w
            yRightBottom = yLeftTop + h

            cv.circle(frame, (xLeftTop, yLeftTop), 5, (0, 255, 0))
            cv.circle(frame, (xRightBottom, yRightBottom), 5, (0, 255, 0))

            cv.rectangle(frame, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom),
                         (0, 255, 0))
            label = "face: %.4f" % confidence
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv.rectangle(frame, (xLeftTop, yLeftTop - labelSize[1]),
                                (xLeftTop + labelSize[0], yLeftTop + baseLine),
                                (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (xLeftTop, yLeftTop),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        return frame

if __name__ == '__main__':
    import sys


    if len(sys.argv) == 1:
        print("""usage:%s [camID | moviefile])
        detect faces in camera or in moviefile.
        """ % sys.argv[0])
        print("Requirement OpenCV 3.4")
        print("cv.__version__", cv.__version__)
        exit()


    confThreshold = 0.5
    detector = ResnetFaceDetector(confThreshold)

    src = sys.argv[1]

    mproccessor = ResnetFaceMovieProcessor(src)
    mproccessor.detector = detector
#    mproccessor.setOutdir("../resnetFace_output")
    mproccessor.setOutdir("")
    mproccessor.process(confThreshold)

