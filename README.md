# MachineLearning_SC

<h3>Installation</h3>

To set up the environment for this project, follow these steps:

1. **Install Conda**  
   Make sure you have Conda installed on your system. You can download it from [Miniconda](https://www.anaconda.com/download) or [Anaconda](https://www.anaconda.com/).

2. **Create the Environment**  
   Use the provided `environment.yml` file to create the Conda environment:
   ```bash
   conda env create -f environment.yml
    ```
- for some packages like tensorflow you have to follow the instructions on their website [Tensorflow](https://www.tensorflow.org/install)

<h3>Commands</h3>
<details>

```bash
python screen_record.py
```
 starts screen recording mode, press `p` to start saving screenshots and `p` again pause it and set new time period. Press `o` to end the script.

- `info` get a quick overview of the amount of directories and files created
- `help` gets you all the parameters

```bash
python prepare_data.py
```
will automatically detect what raw folders are not yet calculated and calculates them. Afterwards the final data is available in data/processed  

</details>



<h3>Notebooks</h3>


<details>

**`model_test_amon.ipynb`** in this notebook you find an example of the correct data extraction and an example with a simple CNN model. 


</details>























<h3>Project structure</h3>
<details>

- this is not up to date 

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         machinelearning_sc and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── machinelearning_sc   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes machinelearning_sc a Python module (not yet)
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

</details>
