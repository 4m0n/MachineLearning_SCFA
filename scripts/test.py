
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os
from pathlib import Path
import cv2

class Colors:
    @staticmethod
    def red():
        return (239, 8, 8)

    @staticmethod
    def darkred():
        return (148, 20, 33)

    @staticmethod
    def orange():
        return (255, 134, 57)

    @staticmethod
    def darkorange():
        return (181, 101, 24)

    @staticmethod
    def darkgold():
        return (165, 150, 0)

    @staticmethod
    def yellow():
        return (255, 251, 0)

    @staticmethod
    def lime():
        return (156, 219, 0)

    @staticmethod
    def green():
        return (66, 190, 66)

    @staticmethod
    def darkgreen():
        return (41, 138, 82)

    @staticmethod
    def teal():
        return (41, 77, 74)

    @staticmethod
    def darkteal():
        return (41, 77, 74) #idk

    @staticmethod
    def blue():
        return (66, 109, 239)

    @staticmethod
    def darkblue():
        return (41, 40, 231)

    @staticmethod
    def purple():
        return (90, 0, 165)

    @staticmethod
    def violet():
        return (148, 97, 255)

    @staticmethod
    def turkis():
        return (99, 255, 206)

    @staticmethod
    def white():
        return (255, 255, 255)

    @staticmethod
    def gray():
        return (99, 109, 123)

    @staticmethod
    def magenta():
        return (255, 138, 255)

    @staticmethod
    def pink():
        return (255, 48, 255)
class Colors2:
    @staticmethod
    def red():
        return (240, 41, 40)

    @staticmethod
    def darkred():
        return (150, 48, 58)

    @staticmethod
    def orange():
        return (255, 135, 58)

    @staticmethod
    def darkorange():
        return (182, 109, 51)

    @staticmethod
    def darkgold():
        return (165, 152, 45)

    @staticmethod
    def yellow():
        return (255, 252, 48)

    @staticmethod
    def lime():
        return (157, 220, 43)

    @staticmethod
    def green():
        return (67, 191, 67)

    @staticmethod
    def darkgreen():
        return (88, 139, 86)

    @staticmethod
    def teal():
        return (64, 91, 88) #idk

    @staticmethod
    def blue():
        return (67, 110, 240)

    @staticmethod
    def darkblue():
        return (88, 80, 232)

    @staticmethod
    def purple():
        return (105, 74, 166)

    @staticmethod
    def violet():
        return (149, 98, 255)

    @staticmethod
    def turkis():
        return (111, 255, 207)

    @staticmethod
    def white():
        return (255, 255, 255)

    @staticmethod
    def gray():
        return (99, 109, 123)

    @staticmethod
    def magenta():
        return (255, 138, 255)

    @staticmethod
    def pink():
        return (255, 48, 255)
    @staticmethod
    def eye():
        return (240, 236, 240)

parent_dir = Path.cwd().parent
# 4 v 4
file_path = parent_dir / "data" / "raw" / "screenshots" / "Session_5s_2025-09-16_15-36-40" / "screenshot_20250916_154109.png"
# 6 v 6
#file_path = parent_dir / "data" / "raw" / "screenshots" / "Session_5s_2025-09-11_02-44-10" / "screenshot_20250911_024441.png"
# 6v6 mid game
file_path = parent_dir / "data" / "raw" / "screenshots" / "Session_5s_2025-09-11_02-44-10" / "screenshot_20250911_025206.png"

# big game
#file_path = parent_dir / "data" / "raw" / "screenshots" / "Session_5s_2025-09-17_16-10-25" / "screenshot_20250917_161057.png"

# game with problems
file_path = parent_dir / "data" / "raw" / "screenshots" / "Session_5s_2025-09-17_20-51-31" / "screenshot_20250917_205220.png"


result = ocr.predict(
    input=str(file_path))

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")


exit()

#pfad = r"pictures/leaderboardfull.png"
pfad = str(file_path)

def get_color(image):
    colors2 = Colors2()
    black_mask = np.all(image == [0, 0, 0], axis=-1)

    non_black_pixels = image[~black_mask]
    #avg_color = np.mean(non_black_pixels, axis=0).astype(np.uint8)
    
    brightest_color = np.max(non_black_pixels, axis=0).astype(np.uint8)
    color_name = "unknown"
    
    # versuch 2 -> euklidische distanz
    cs = []
    for func_name in dir(colors2):
        if func_name.startswith("__"):
            continue
        cs.append((func_name, getattr(colors2, func_name)()))
    
    min_value_score = 255**3
    min_value = []
    for c in cs:
        dist = np.array(brightest_color) - np.array(c[1])
        temp = 0
        for d in dist:
            temp = temp + d*d
        if min_value_score > temp:
            min_value_score = temp
            color_name = c[0]
            min_value = c


    new_image = np.zeros_like(image)
    new_image[:, :] = brightest_color 

    return new_image,color_name

def get_number(image):
    #white = np.array([255, 255, 255])
    white = np.array([1, 1, 1])
    diff = np.abs(image - white)  # Differenz zu Weiß berechnen
    mask = np.all(diff < 0.6, axis=-1)  # Prüfen, ob alle Kanäle innerhalb des Thresholds liegen

    # Neues Bild erstellen: Weiße Pixel bleiben, andere werden schwarz
    result = np.zeros_like(image)  # Alles schwarz
    result[mask] = white 
    result = (result * 255).astype(np.uint8)
    height, width = result.shape[:2]
    new_width = int(width * 3)
    new_height = int(height * 3)
    #result = cv2.resize(result, (new_width, new_height), interpolation=cv2.INTER_CUBIC) # sehr wichtig sonst gehts nicht
    
    scale_factor = 4
    result = cv2.resize(result, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    

    result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    _, result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_not(result)
    #result = cv2.GaussianBlur(result, (3,3), 0)
    #result = cv2.blur(result,(5,5))
    result = ocr.predict(
        input=result)  
    print(result)
    return result,text
    

img = mpimg.imread(pfad)
height, width, _ = img.shape

next = abs((1-0.0689) - 0.9163)
num = 2#17
colors = Colors()
cs = []
for func_name in dir(colors):
    if func_name.startswith("__"):
        continue
    cs.append((func_name, getattr(colors, func_name)()))

fig, axes = plt.subplots(4,num,  figsize=(14, 5))
for i in range(num):
    # Players
    crop_top = int(height * (0.0689+(next*i)))  #  oben
    crop_bottom = int(height * (1 - (0.9163-(next*i))))  #  unten
    crop_left = int(width * 0.768)  #  links
    crop_right = int(width * (1 - 0.225))  #  rechts
    
    # Power
    crop_top2 = int(height * (0.0689+(next*i)))  #  oben
    crop_bottom2 = int(height * (1 - (0.9163-(next*i))))  #  unten
    crop_left2 = int(width * 0.85)  #  links
    crop_right2 = int(width * (1 - 0.126))  #  rechts
    
    
    # Bild zuschneiden
    players = img[crop_top:crop_bottom, crop_left:crop_right]
    power = img[crop_top2:crop_bottom2, crop_left2:crop_right2]

    def conv_img(cropped_img,alpha = 0.3):
        scaled_img = (cropped_img * 255).astype(np.uint8)
        new_image = cv2.convertScaleAbs(scaled_img, alpha=alpha, beta=1)
        return new_image

    players = conv_img(players, alpha=1)
    color_pic,color_name = get_color(players)
    if color_name == "eye":
        break
    
    axes[0, i].imshow(players)
    axes[0, i].set_title(f"P {i+1}")
    axes[0, i].axis("off")  # Achsen ausblenden
    
    #power = conv_img(power)
    power,text = get_number(power)
    axes[1, i].imshow(power)
    if text ==[]:
        axes[1, i].set_title(f"none")
    else:
        axes[1, i].set_title(f"{text[0][1]}")
    axes[1, i].axis("off")  # Achsen ausblenden
    
    # Bild in der zweiten Zeile anzeigen
    axes[2, i].imshow(color_pic)
    axes[2, i].set_title(f"{color_name}\n{color_pic[0,0]}",fontsize=8)
    axes[2, i].axis("off")  # Achsen ausblenden
    
    # alle farben aus leaderboard
    new_image = np.zeros_like(players)
    new_image[:, :] = cs[i][1]
    axes[3, i].imshow(new_image)
    axes[3, i].set_title(f"{cs[i][0]}")
    axes[3, i].axis("off")  # Achsen ausblenden
    
plt.tight_layout()
plt.show()