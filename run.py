import cv2
import os.path
from compare_files import compare_files

folder = "C:\\Users\\asmolinski\\Zachodniopomorski Uniwersytet Technologiczny w Szczecinie\\Paweł Forczmański - database\\#004\\warp\\"
thermal_img = '#000_ImgFLIR2'
visual_img = '#000_ImgCANON2'
type = ".PNG"

#print(os.path.isfile(folder+thermal_img+type))

compare_files(cv2.imread(thermal_img+type,cv2.IMREAD_GRAYSCALE), cv2.imread(visual_img+type,cv2.IMREAD_GRAYSCALE))