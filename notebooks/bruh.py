import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parents[0]))
import DataLoader

import matplotlib.pyplot as plt
stats = DataLoader.LoadCNN(return_all=True)
time_plot = stats["time"].dropna()
#plt.hist(time_plot, bins=30)  # bins kannst du anpassen
plt.scatter([1,2,3,4],[2,3,4,5])
plt.xlabel("time")
plt.ylabel("Anzahl")
plt.title("Histogramm der Zeitwerte")
plt.show()