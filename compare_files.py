import config
from imutils import face_utils
from draw_shape import draw_shape
from math import sqrt

def compare_files(thermal_image, visible_image):

    thermal_shape = draw_shape(thermal_image, config.thermal_cascade_path, config.thermal_predictor_path)
    visual_shape = draw_shape(visible_image, config.visual_cascade_path, config.visual_predictor_path)
    #convert coordinates to np-array
    thermal_shape = face_utils.shape_to_np(thermal_shape)
    visual_shape = face_utils.shape_to_np(visual_shape)
    
    #Calculate distances between all 68 points
    distanceSum = 0
    minimalDistance = float('inf')
    maximalDistance = float('-inf')
    for (i, (x, y)) in enumerate(thermal_shape):
        distanceBetweenPoints = sqrt((x-visual_shape[i][0])**2+(y-visual_shape[i][1])**2)
        distanceSum = distanceSum + distanceBetweenPoints
        if maximalDistance < distanceBetweenPoints:
            maximalDistance = distanceBetweenPoints
        if minimalDistance > distanceBetweenPoints:
            minimalDistance = distanceBetweenPoints

    return [distanceSum/(i+1), minimalDistance, maximalDistance]
