import config
import cv2
import dlib

def draw_shape(img, cascade, predictor):

    predictor = dlib.shape_predictor(predictor)
    cascade_face = cv2.CascadeClassifier(cascade)
    
    #===========automatic face mark
    initBB = cascade_face.detectMultiScale(img, 1.05, 1, minSize=(100,100))
    box = tuple(initBB[0])
    #===========manual face mark
    #box = cv2.selectROI("Mark face", img, fromCenter=False, showCrosshair=True)

    shape = predictor(img, dlib.rectangle(int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])))

    #===========Display prediction
    # win = dlib.image_window()
    # win.add_overlay(dlib.rectangle(int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])))
    # win.add_overlay(shape)
    # win.set_image(img)
    #dlib.hit_enter_to_continue()

    return shape
