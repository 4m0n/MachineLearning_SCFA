import os
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
import numpy as np
import typer
from typing import Optional
import shutil
from tqdm import tqdm
import pandas as pd
import matplotlib.image as mpimg
import ast

app = typer.Typer()

# colors from pre selection
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
        return (41, 77, 74)

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

# leaderboard colors
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



# ========= Get Power and Players =========


def get_color(image):
    colors2 = Colors2()
    black_mask = np.all(image == [0, 0, 0], axis=-1)

    non_black_pixels = image[~black_mask]
    #avg_color = np.mean(non_black_pixels, axis=0).astype(np.uint8)
    
    brightest_color = np.max(non_black_pixels, axis=0).astype(np.uint8)
    color_name = "unknown"
    
    cs = []
    for func_name in dir(colors2):
        if func_name.startswith("__"):
            continue
        cs.append((func_name, getattr(colors2, func_name)()))
    
    min_value_score = 255**3
    for c in cs:
        dist = np.array(brightest_color) - np.array(c[1])
        temp = 0
        for d in dist:
            temp = temp + d*d
        if min_value_score > temp:
            min_value_score = temp
            color_name = c[0]



    return color_name
# extract time from image
def get_time(result):

    result = (result * 255).astype(np.uint8)
    scale_factor = 4
    result = cv2.resize(result, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    text = ocr.predict(result)  
    text = text[0]["rec_texts"]
    i = 0
    while i < len(text):
        if any(char not in ['0','1','2','3','4','5','6','7','8','9',':'] for char in text[i]):
            return np.nan
        i += 1
    if text == []:
        text = np.nan
    else:
        text = text[0]
    return text
# extract power number from image 
def get_number(image):
    white = np.array([1, 1, 1])
    diff = np.abs(image - white)  # Differenz zu Weiß berechnen
    mask = np.all(diff < 0.6, axis=-1)  # Prüfen, ob alle Kanäle innerhalb des Thresholds liegen

    result = np.zeros_like(image)  # Alles schwarz
    result[mask] = white 
    result = (result * 255).astype(np.uint8)
    height, width = result.shape[:2]

    scale_factor = 4
    result = cv2.resize(result, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    

    result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    _, result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_not(result)
    #result = cv2.GaussianBlur(result, (3,3), 0)
    #result = cv2.blur(result,(5,5))
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    text = ocr.predict(result)  
    text = text[0]["rec_texts"]
    i = 0
    while i < len(text):
        if any(char not in ['0','1','2','3','4','5','6','7','8','9','k','m','.'] for char in text[i]):
            return np.nan
        i += 1
    if text == []:
        text = np.nan
    else:
        text = text[0]
    return text
# load all informtions from image and save them in a dict
def preprocess_image_power(pfad, new_row):

    img = mpimg.imread(pfad)
    height, width, _ = img.shape

    next = abs((1-0.0689) - 0.9163)
    num = 17
    colors = Colors()
    cs = []
    for func_name in dir(colors):
        if func_name.startswith("__"):
            continue
        cs.append((func_name, getattr(colors, func_name)()))


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
        color_name = get_color(players)
        if color_name == "eye":
            break
        
        text = get_number(power)
        new_row[color_name] = text

    # Time
    crop_top3 = int(height * (0.035))  #  oben
    crop_bottom3 = int(height * (1 - (0.95)))  #  unten
    crop_left3 = int(width * 0.776)  #  links
    crop_right3 = int(width * (1 - 0.194))  #  rechts
    time = img[crop_top3:crop_bottom3, crop_left3:crop_right3]
    time = get_time(time)
    new_row["time"] = time
    return new_row
        
def full_numbers(data):
    i = 0
    while i < len(data):
        for col in data.columns:
            if isinstance(data.at[i, col], str):
                if 'k' in data.at[i, col]:
                    try:
                        num = float(data.at[i, col].replace('k', '')) * 1000
                        data.at[i, col] = float(num)
                    except ValueError:
                        data.at[i, col] = np.nan
                elif 'm' in data.at[i, col]:
                    try:
                        num = float(data.at[i, col].replace('m', '')) * 1000000
                        data.at[i, col] = float(num)
                    except ValueError:
                        data.at[i, col] = np.nan
                elif ":" in data.at[i, col]:
                    try:
                        h, m, s = map(int, data.at[i, col].split(":"))
                        data.at[i, col] = int(h * 3600 + m * 60 + s)
                    except ValueError:
                        return np.nan
                        
            elif pd.isna(data.at[i, col]):
                data.at[i, col] = np.nan
        i += 1
    return data
    
    
            
# =========== Get Pixels ===========

# saves pictures in different folders for each player color
def save_color_pictures(players, files, ouput_dir, direct):
    # create folder -> delete if it exists to make easy reacalc
    general_dir = ouput_dir / direct

    if os.path.exists(general_dir):
        shutil.rmtree(general_dir)
    os.makedirs(general_dir)
    
    # ==== Stats file ====
    
    colors = Colors()
    color_names = []
    for func_name in dir(colors):
        if func_name.startswith("__"):
            continue
        color_names.append(func_name)
    stats = pd.DataFrame(columns=["Frame"] + color_names)
    info_power = pd.DataFrame(columns=["Frame","time"] + color_names)
    # ====================
    
    
    first = files[0]
    extract_colors = []
    for func_name in dir(colors):
        if func_name.startswith("__"):
            continue
        extract_colors.append(func_name)
        
    for file in tqdm(files,desc = "Calculating Scores",leave=False):    
        base_name = os.path.basename(file)
        new_power_row = {"Frame": base_name[:-4]}
        new_power_row = preprocess_image_power(file,new_power_row)
        info_power = pd.concat([info_power, pd.DataFrame([new_power_row])], ignore_index=True) 
        
        
    info_power = full_numbers(info_power)
    filtered_df = info_power.dropna(axis=1, how='all')
    valid_list = filtered_df.columns.tolist() 
    i = 0
    while i < len(extract_colors): # speeds things up a lot (more if less players)
        if extract_colors[i] not in valid_list:
            extract_colors.pop(i)
            i-=1
        i+=1 
           
               
    for file in tqdm(files,desc = "Processing Screenshots",leave=False):
        img = subtract_pics(first,file)
        base_name = os.path.basename(file)
        new_row = {"Frame": base_name[:-4]}

        for func_name in extract_colors:    
            func = getattr(colors, func_name)
            pic,count = count_color_pixel(img,target_color=func(), tolerance=40,count=False)
            new_row[func_name] = count
            if count > 1000:
                
                color_dir = general_dir / func_name    
                os.makedirs(color_dir, exist_ok=True)
                cv2.imwrite(os.path.join(color_dir, base_name), pic) 
                
                
        
        
            
        stats = pd.concat([stats, pd.DataFrame([new_row])], ignore_index=True)
    
    # ======= Save all Information in one file =======
    """
    Structure of file 
    Frame , time, colors
    frame1,   10, [power, area, more...]
    """
    screens = info_power["Frame"].tolist()
     
    data = pd.DataFrame(columns=["Frame","time"] + color_names) # power, area, more 
    for pic in screens:
        info_power_row = info_power[info_power["Frame"] == pic]
        stats_row = stats[stats["Frame"] == pic]
        new_row = {"Frame": pic}
        for col in info_power_row:
            if col.lower() in ["frame", "time"]:
                continue
            try:
                power = int(info_power_row[col].values[0])
            except:
                power = np.nan
            try:
                area = int(stats_row[col].values[0])
            except:
                area = np.nan
            new_row[col] = [power, area] # here you can add more entries
            
        new_row["time"] = info_power_row["time"].values[0]
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
     
    data.to_csv(general_dir / "stats.csv", index=False)    
    picture_without_background(ouput_dir, direct)
    
    

# gets subtracted picutre and counts pixels of a certain color
def count_color_pixel(img,target_color, tolerance=50, count = False):
    # RGB Farbraum -> problem mit weiss und gelb -> schwarz wird erkannt   


    img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    target_r, target_g, target_b = target_color
    img = img_org.astype(np.int16)
    target_r = np.int16(target_r)
    target_g = np.int16(target_g) 
    target_b = np.int16(target_b)
    

    color_mask = (
        (np.abs(img[:, :, 0] - target_r) < tolerance) &  # Rot
        (np.abs(img[:, :, 1] - target_g) < tolerance) &  # Grün
        (np.abs(img[:, :, 2] - target_b) < tolerance)    # Blau
    )
        
    white_mask = np.all(img_org >= [255, 255, 255], axis=-1)    
    
    
    result_mask = color_mask.copy()
    diff = []
    for shift in range(1, 7):  
        shifted = np.roll(white_mask, -shift, axis=0)
        valid = color_mask[:-shift, :] & shifted[:-shift, :]
        shiftedback = np.roll(valid, shift, axis=0)
        result_mask[:-shift, :] |= shiftedback 
        diff.append(np.count_nonzero(result_mask) - np.count_nonzero(color_mask))
    color_mask = result_mask
    pixel_count = np.count_nonzero(color_mask)
    if not count:
        result = np.zeros_like(img_org)
        result[color_mask] = img_org[color_mask]
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result_bgr,pixel_count

    return pixel_count
    
# subtracts two picutres and shows original pixel values of masked second picture 
def subtract_pics(start_pfad, test_pfad):
    img1 = cv2.imread(start_pfad)
    img2 = cv2.imread(test_pfad)

    diff = cv2.absdiff(img1, img2)

    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    #count = cv2.countNonZero(mask)
    # Wende die Maske auf Bild 2 an, um nur die unterschiedlichen Pixel zu behalten
    result = cv2.bitwise_and(img2, img2, mask=mask)
    return result    

# save picture without background -> combine all loaded pictures
def picture_without_background(ouput_dir, direct):
    general_dir = ouput_dir / direct
    subBackground = general_dir / "subBackground"
    if os.path.exists(subBackground):
        shutil.rmtree(subBackground)
    os.makedirs(subBackground)
    
    all_image_paths = []
    for folder in general_dir.iterdir():
        if folder.is_dir():
            all_image_paths.append([])
            for file in folder.iterdir():
                if file.is_file(): 
                    all_image_paths[-1].append(str(file))
    
    

    org_folder = Path(f"{config.RAW_DATA_DIR}/screenshots/{direct}")
    org_files = [file for file in org_folder.iterdir() if file.is_file()]
    
    
    for org_file in tqdm(org_files,desc="Picture without Background",leave=False):
        temp_img = cv2.imread(org_file, cv2.IMREAD_COLOR)
        combined_img = np.zeros_like(temp_img)
        for folder in all_image_paths:
            i = 0
            while i < len(folder):
                if folder[i].endswith(org_file.name):
                    add_image = cv2.imread(folder[i], cv2.IMREAD_COLOR)
                    combined_img = cv2.add(combined_img, add_image)
                    break
                i+=1
        cv2.imwrite(subBackground / org_file.name, combined_img)
        
        
    ...
# checks how often pixel of a certain colors show up -> threshold to detect players
def find_players(files):
    files.sort()
    start = files[0]
    pic1 = files[5]
    result = subtract_pics(start,pic1)
    
    colors = Colors()

    player_count = 0
    players = []
    for func_name in dir(colors):
        if func_name.startswith("__"):
            continue
        func = getattr(colors, func_name)
        count = count_color_pixel(result,target_color=func(), tolerance=40,count=True)
        if count > 1000:
            player_count += 1
            players.append(func_name)
    return players

# returns list of files that are not yet proccesed ( execpt recalculate is True )
def list_outputs(input_dir,ouput_dir,recalculate):
    outputs = []
    for file in os.listdir(ouput_dir):
        outputs.append(file)
    inputs = []
    for file in os.listdir(input_dir):
        if file in outputs and not recalculate:
            continue
        inputs.append(file) 
    return inputs



@app.command()
def prepare_data(
    input_dir: Path = typer.Option(Path(f"{config.RAW_DATA_DIR}/screenshots/"), help="Input directory"),
    ouput_dir: Path = typer.Option(Path(f"{config.PROCESSED_DATA_DIR}"), help="Ouput directory"),
    recalculate: bool = typer.Option(False, help="Recalculate all files"),
    praefix: str = typer.Option("screenshot", help="Präfix für Dateinamen")
):
    inputs = list_outputs(input_dir,ouput_dir,recalculate)
    for dir in tqdm(inputs,desc="Processing Sessions"):
        load_dir = input_dir / dir
        files = os.listdir(load_dir)
        if len(files) > 5:
            full_paths = [str(load_dir / file) for file in files]
            #if "Session_5s_2025-09-11_02-44-10" in str(load_dir): # just to test
            #players = find_players(full_paths)
            save_color_pictures("players",full_paths,ouput_dir, dir)
                


if __name__ == "__main__":
    app()
    
    
    
    