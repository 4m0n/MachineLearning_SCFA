import pandas as pd
import numpy as np
from pathlib import Path
import ast
import cv2
import os
from pathlib import Path
from keras.models import load_model
from keras.losses import mean_squared_error
from keras.metrics import MeanSquaredError

import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
    
import config

def LoadCNN():
    pfad = config.PROCESSED_DATA_DIR
    main_path = Path(pfad)

    all_information = pd.DataFrame()

    for session in main_path.iterdir():
        if session.is_dir():
            farb_liste = []
            all_stats = None

            # First find the stats.csv file
            for color in session.iterdir():
                if color.is_file() and color.name == "stats.csv":
                    all_stats = pd.read_csv(color)
                    break  # Assuming there's only one stats.csv per directory

            if all_stats is None:
                print(f"No stats.csv found in directory {session}")
                continue  # Skip this session if stats.csv is not found

            # Check if 'Frame' column exists
            if 'Frame' not in all_stats.columns:
                print(f"Warning: 'Frame' column not found in {session}")
                continue

            # Check if 'time' column exists
            if 'time' not in all_stats.columns:
                print(f"Warning: 'time' column not found in {session}")
                continue

            # Collect color columns (excluding 'Frame' and 'time')
            color_columns = [col for col in all_stats.columns if col not in ['Frame', 'time'] and col != "subBackground"]

            data = pd.DataFrame()
            pfad_list = []
            time_list = []
            stats_list = []

            for i in range(len(all_stats)):
                frame = all_stats["Frame"].iloc[i]
                time_temp = all_stats["time"].iloc[i]

                for col in color_columns:
                    pfad_list.append(f"{str(session)}/{col}/{frame}.png")
                    time_list.append(time_temp)
                    value = str(all_stats[col].iloc[i])
                    if "nan" in value:
                        value = value.replace("nan", "0")
                    try:
                        stats_list.append(ast.literal_eval(value))
                    except Exception as e:
                        print(f"Error in {col} at {value} - Dir {session} {frame}: {e}")
                        stats_list.append([-1, -1])  # Default values for power and area

            if not stats_list:
                continue  # Skip if no valid data was collected

            stats_array = np.array(stats_list)
            # Ensure stats_array has at least 2 columns
            if stats_array.shape[1] < 2:
                print(f"Warning: insufficient data in stats list for {session}")
                continue

            data = pd.DataFrame({
                "path": pfad_list,
                "time": time_list,
                "power": stats_array[:, 0],
                "area": stats_array[:, 1]
            })

            all_information = pd.concat([all_information, data], ignore_index=True)

    all_information = all_information.dropna(subset=["power"])
    all_information = all_information[["path"]]
    return all_information

def rgb_to_binary(image_array):
    # Check if all channels (R, G, B) are 0 for each pixel
    is_black = np.all(image_array == 0, axis=2)
    # Convert the boolean array to integer: 1 if not black, 0 if black
    binary_array = np.where(is_black, 0, 1).astype(np.uint8)
    # Reshape to (x, y, 1)
    binary_array = binary_array.reshape(image_array.shape[0], image_array.shape[1], 1)
    return binary_array


model = load_model('units/categories.keras', custom_objects={'mse': mean_squared_error})

names = ['air', 'land', 'naval', 'building', 'tier1', 'tier2', 'tier3']

while True:
    paths = LoadCNN()

    true_paths = []
    for i in range(len(paths['path'])):
        if os.path.exists(paths['path'][i]):
            true_paths.append(paths['path'][i])
    del paths

    if len(true_paths) == 0:
        break

    print(len(true_paths))
    
    for path in true_paths:
        filepath = Path(path.replace('processed', 'KNN').replace('.png', '.csv'))
        fileparent = os.path.dirname(filepath)
        if not os.path.exists(fileparent):
            os.makedirs(fileparent)  
        if not os.path.exists(filepath):
            image = cv2.imread(path)
            imagebw = rgb_to_binary(image)
            pred = model.predict(imagebw.reshape(1, 1080, 1920, 1))
            np.savetxt(filepath, np.stack([names, np.array(pred).squeeze()], axis=1), delimiter=',', fmt='%s')

    