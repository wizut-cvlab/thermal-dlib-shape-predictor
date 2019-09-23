import config
from scripts.draw_shape import draw_shape
from math import sqrt


def compare_files(thermal_image, visible_image):

    [visual_shape, box] = draw_shape(
        visible_image, config.visual_cascade_path, config.visual_predictor_path
    )
    [thermal_shape, box] = draw_shape(
        thermal_image,
        config.thermal_cascade_path,
        config.thermal_predictor_path,
        box=box,
    )
    if thermal_shape == [] or visual_shape == []:
        return [[config.detection_error_text], []]

    # Calculate distances between all 68 points
    distanceSum = 0
    minimalDistance = float("inf")
    maximalDistance = float("-inf")
    pointsDistances = []
    for (i, (x, y)) in enumerate(thermal_shape):
        distanceBetweenPoints = sqrt(
            (x - visual_shape[i][0]) ** 2 + (y - visual_shape[i][1]) ** 2
        )
        pointsDistances.append(distanceBetweenPoints)
        distanceSum = distanceSum + distanceBetweenPoints
        if maximalDistance < distanceBetweenPoints:
            maximalDistance = distanceBetweenPoints
        if minimalDistance > distanceBetweenPoints:
            minimalDistance = distanceBetweenPoints

    return [[distanceSum / (i + 1), minimalDistance, maximalDistance], pointsDistances]
