import cv2
import os
import config
from scripts.compare_files import compare_files

# For each Folder in ./images
for subFolderName in os.listdir(config.image_path):
    # For each Image in subFolder
    for imageFile in os.listdir(config.image_path + subFolderName):
        # If Thermal image
        if imageFile.find(config.thermal_image_suffix) > -1:
            imageFileName = imageFile[0:4]
            thermalImagePath = (
                config.image_path
                + subFolderName
                + "\\"
                + imageFileName
                + config.thermal_image_suffix
                + config.image_extention
            )
            visualImagePath = (
                config.image_path
                + subFolderName
                + "\\"
                + imageFileName
                + config.visual_image_suffix
                + config.image_extention
            )
            [summary, distances, normalizationParams] = compare_files(
                cv2.imread(thermalImagePath, cv2.IMREAD_GRAYSCALE),
                cv2.imread(visualImagePath, cv2.IMREAD_GRAYSCALE),
            )
            print(subFolderName, imageFileName, normalizationParams)
