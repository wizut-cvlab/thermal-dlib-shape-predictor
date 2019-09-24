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
        return [[config.detection_error_text], [], []]

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

    outerEyeCornersDistance = distanceBetweenPoints = sqrt(
        (thermal_shape[45][0] - thermal_shape[36][0]) ** 2
        + (thermal_shape[45][1] - thermal_shape[36][1]) ** 2
    )

    boxDiagonal = sqrt((box[2]) ** 2 + (box[3]) ** 2)

    return [
        [distanceSum / (i + 1), minimalDistance, maximalDistance],
        pointsDistances,
        [outerEyeCornersDistance, boxDiagonal],
    ]
