# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:13:32 2018

@author: pawel
"""

import numpy as np
import matplotlib as mpl
import cv2
import dlib

from imutils import face_utils
import argparse
import imutils
import time
import sys

cascade_face = cv2.CascadeClassifier(
    "..\\cascades\\haarcascade_frontalface_default.xml"
)
cascade_thr = cv2.CascadeClassifier("..\\cascades\\thr_haar_cascade.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "..\\predictors\\shape_predictor_68_face_landmarks.dat"
)
predictor_thr = dlib.shape_predictor("..\\predictors\\predictor4.dat")

writer = cv2.VideoWriter(
    "output22.avi", cv2.VideoWriter_fourcc(*"XVID"), 5, (800, 300), 1
)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def bb_to_rect(rr):
    print(type(rr))
    rect2 = dlib.rectangle(
        left=rr[0], top=rr[1], right=rr[2] + rr[0], bottom=rr[3] + rr[1]
    )
    return rect2


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


for i in range(1, 101):
    # kat="thr_dataset\\jasinski\\#" + str(i).zfill(3)
    kat = "..\\images\\#" + str(i).zfill(3)
    for j in range(0, 5):
        plik_vis = "\\#" + str(j).zfill(3) + "_imgCANON2.PNG"
        plik_thr = "\\#" + str(j).zfill(3) + "_imgFLIR2.PNG"
        # plik="C:\\Users\\pawel\\Documents\\MATLAB\\bazy\\wizut\\#001\\#001_canon.JPG"
        color = cv2.imread(kat + plik_vis, 0)
        if color is None:
            continue

        thr = cv2.imread(kat + plik_thr, 0)
        if thr is None:
            continue

        # cv2.imshow('thr', thr)
        # cv2.waitKey(1)
        # cv2.imshow('color', color)
        # cv2.waitKey(1)

        # gray = cv2.resize(color, (200, 300))
        # gray_thr = cv2.resize(thr, (200, 300))
        # gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        # print(plik_vis)
        gray = color.copy()
        gray_thr = thr.copy()

        faces = cascade_face.detectMultiScale(gray, 1.05, 1)
        rects = detector(gray, 1)
        faces_thr = cascade_thr.detectMultiScale(gray_thr, 1.05, 1, minSize=(100, 100))
        rects_thr = detector(gray_thr, 1)

        color = np.zeros((gray.shape[0], gray.shape[1], 3), dtype="uint8")
        color[:, :, 0] = gray
        color[:, :, 1] = gray
        color[:, :, 2] = gray

        color_thr = np.zeros((gray_thr.shape[0], gray_thr.shape[1], 3), dtype="uint8")
        color_thr[:, :, 0] = gray_thr
        color_thr[:, :, 1] = gray_thr
        color_thr[:, :, 2] = gray_thr

        for (x, y, w, h) in faces:
            cv2.rectangle(color, (x, y), ((x + w), (y + h)), (230, 10, 10), 2)

        for i in rects:
            rects2 = i
            cv2.rectangle(
                color,
                (rects2.left(), rects2.top()),
                ((rects2.left() + rects2.width()), (rects2.top() + rects2.height())),
                (130, 130, 130),
                2,
            )

        for (x, y, w, h) in faces_thr:
            cv2.rectangle(color_thr, (x, y), ((x + w), (y + h)), (10, 10, 230), 2)

        for i in rects_thr:
            rects2 = i
            cv2.rectangle(
                color_thr,
                (rects2.left(), rects2.top()),
                ((rects2.left() + rects2.width()), (rects2.top() + rects2.height())),
                (230, 130, 130),
                2,
            )

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(color, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(color, (x, y), 1, (0, 0, 255), -1)
                # cv2.circle(color_thr, (x, y), 1, (0, 0, 255), -1)
                # print([x,y])

        # for dlib thr predictor
        # for (i, face_thr) in enumerate(faces_thr):
        for (i, rect) in enumerate(rects):

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            # print(face_thr)
            # print(type(face_thr))

            # rect2=bb_to_rect(face_thr)
            # rect2=bb_to_rect(rect_thr)
            # cv2.normalize(gray_thr, gray_thr, 0, 255, cv2.NORM_MINMAX)
            shape_thr = predictor_thr(gray_thr, rect)
            shape_thr = face_utils.shape_to_np(shape_thr)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(color_thr, (x, y), (x + w, y + h), (10, 10, 230), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape_thr:
                # cv2.circle(color, (x, y), 1, (0, 0, 255), -1)
                cv2.circle(color_thr, (x, y), 1, (255, 0, 255), -1)
            # print([x,y])
        # show the face number
        # cv2.putText(color, "OpenCV #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(
            color, "OpenCV vis", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        cv2.putText(
            color_thr,
            "OpenCV thr",
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 10, 230),
            1,
        )
        cv2.putText(
            color, "Dlib vis", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 10, 10), 1
        )
        cv2.putText(
            color_thr,
            "Dlib vis",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 130, 130),
            1,
        )

        color_s = cv2.resize(color, (400, 300))
        thr_s = cv2.resize(color_thr, (400, 300))
        stacked = np.hstack((color_s, thr_s))
        cv2.imshow("...", stacked)
        # cv2.waitKey(0)
        writer.write(stacked)
        #
        # cv2.WriteFrame(writer, gray)
        # time.sleep(1)
        # cv2.destroyAllWindows()
        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            writer.release()
            sys.exit()
        if k == 32:
            time.sleep(10)

writer.release()
cv2.destroyAllWindows()
