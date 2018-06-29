# -*- coding: utf-8 -*-
import numpy as np
import cv2

def sideBySide(movie0, movie1, outMovie):
    """
    generate side by side movie.

    """

    cap0 = cv2.VideoCapture(movie0)
    cap1 = cv2.VideoCapture(movie1)

    vout = None

    FRAME_RATE = 30

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        shape0 = frame0.shape
        shape1 = frame1.shape

        if vout is None:
            cols = shape0[1]+shape1[1]
            rows = max(shape0[0], shape1[0])
            vout = cv2.VideoWriter(outMovie, \
                                  cv2.VideoWriter_fourcc(*'MJPG'), \
                                  FRAME_RATE, \
                                  (cols, rows))

        newFrame = np.hstack((frame0, frame1))

        vout.write(newFrame)

    vout.release()
    cap0.release()
    cap1.release()


if __name__ == "__main__":
    movie0 = '/home/papa/.gvfs/smb-share:server=nas.dev.groove-x.io,share=image_dataset/detection_result/resnetssd/banzai_resized_out.avi'
    movie1 = '/home/papa/gitHub/track_result/banzai_resized_out.avi'

    outMovie = "junk.avi"

    movie0 = '/home/papa/.gvfs/smb-share:server=nas.dev.groove-x.io,share=image_dataset/detection_result/resnetssd/papakai_20180407_resized_out.avi'
    movie1 = '/home/papa/gitHub/track_result/papakai_20180407_resized_out.avi'

    outMovie = "papakai_junk.avi"

    import sys
    import os
    
    if len(sys.argv) == 4:
        movie0, movie1, outMovie = sys.argv[1:]
    elif len(sys.argv) == 3:
        movie0, movie1 = sys.argv[1:]
        base0 = os.path.splitext(os.path.basename(movie0))[0]
        base1 = os.path.splitext(os.path.basename(movie1))[0]
        outMovie = "%s_%s.avi" % (base0, base1)
    
    sideBySide(movie0, movie1, outMovie)