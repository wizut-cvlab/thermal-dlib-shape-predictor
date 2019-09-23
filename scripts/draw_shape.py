import config
import cv2
import dlib
from imutils import face_utils


def draw_shape(img, cascade, predictor, show=False, manualMarking=False, box=None):

    # ===========face mark from other source
    if box == None:
        # ===========automatic face mark
        if manualMarking == False:
            initBB = cascade.detectMultiScale(img, 1.05, 1, 0, (75, 75))
            if len(initBB) < 1:
                return [[], None]

            box = tuple(initBB[0])
        # ===========manual face mark
        else:
            box = cv2.selectROI("Mark face", img, fromCenter=False, showCrosshair=True)
            print(box)

    shape = predictor(
        img,
        dlib.rectangle(
            int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])
        ),
    )

    # ===========Display prediction
    if show:
        win = dlib.image_window()
        win.add_overlay(
            dlib.rectangle(
                int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])
            )
        )
        win.add_overlay(shape)
        win.set_image(img)
        dlib.hit_enter_to_continue()
        cv2.destroyAllWindows()

    return [face_utils.shape_to_np(shape), box]
