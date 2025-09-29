import pandas as pd
import numpy as np
import config
from pathlib import Path
import ast

def LoadCNN():
    """
    INPUT:
    None
    
    OUTPUT:
    - Dataframe with only paths 
    
                                                            path
    0      E:\Coding\Python\MachineLearning_SCFA\data\pro...
    1      E:\Coding\Python\MachineLearning_SCFA\data\pro...
    2      E:\Coding\Python\MachineLearning_SCFA\data\pro...
    """
    
    pfad = config.PROCESSED_DATA_DIR 
    main_path = Path(pfad)
    
    all_stats = pd.DataFrame()
    all_information = pd.DataFrame()
    for session in main_path.iterdir():
        if session.is_dir():
            farb_liste = []
            for color in session.iterdir():
                if color.is_file() and color.name == "stats.csv":
                    all_stats = pd.read_csv(color)
                else:
                    if color.name != "subBackground":
                        farb_liste.append(color.name)
                
            data = pd.DataFrame()
            
            i = 0
            time_temp = 0
            time_list = []
            pfad_list = []
            stats_list = []
            while i < len(all_stats):
                frame = "unkown"
                for col in all_stats.columns:
                    frame = all_stats["Frame"].iloc[i]
                    time_temp = all_stats["time"].iloc[i]
                    if col == "Frame":
                        continue
                    elif col == "time":
                        continue
                    elif col not in farb_liste:
                        continue
                    else:    
                        pfad_list.append(f"{str(session)}\{col}\{frame}.png")
                        time_list.append(time_temp)
                        if col in all_stats.columns.values:
                            value = str(all_stats[col].iloc[i])
                            if "nan" in value:
                                value = value.replace("nan", "0")
                            try:
                                stats_list.append(ast.literal_eval(value))
                            except:
                                print(f"Error in {col} at {value} - Dir {session} {frame}")
                        else:
                            stats_list.append(-1)


                        
                i += 1
            """
            Structure of file 
            Frame , time, colors
            frame1,   10, [power, area, more...]
            """
            stats_list = np.array(stats_list)
            data = pd.DataFrame({
                "path": pfad_list,
                "time": time_list,
                "power": stats_list[:,0],
                "area": stats_list[:,1]
            })
            

            all_information = pd.concat([all_information, data], ignore_index=True)           
              
            

    all_information = all_information.dropna(subset=["power"])
    all_information = all_information[["path"]]
    return all_information
    
    
    
    
def loadKNN():
    """
    INPUT:
    
    
    OUTPUT:
    
    """
    ...