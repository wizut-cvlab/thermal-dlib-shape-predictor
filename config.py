import dlib
import cv2

visual_image_suffix = "_ImgCANON2"
thermal_image_suffix = "_ImgFLIR2"
image_extention = ".PNG"
image_path = "images\\"

thermal_predictor_path = dlib.shape_predictor("predictors\\predictor4.dat")
thermal_cascade_path = cv2.CascadeClassifier(
    "cascades\\haarcascade_frontalface_default.xml"
)

visual_cascade_path = cv2.CascadeClassifier(
    "cascades\\haarcascade_frontalface_default.xml"
)
visual_predictor_path = dlib.shape_predictor(
    "predictors\\shape_predictor_68_face_landmarks.dat"
)

detection_error_text = "No face detection"
