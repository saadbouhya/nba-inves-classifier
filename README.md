# NBA Player Classifier for Longevity Prediction

## Overview

This project aims to predict whether NBA players will have a career lasting more than 5 years, serving as a tool for investors interested in player longevity. The classification model uses player statistics to make predictions.

### Contents:

1. **Exploratory Data Analysis (EDA)**:
   - Jupyter Notebook: [EDA.ipynb](src/EDA.ipynb)
   - Description: This notebook explores the dataset containing NBA player statistics, identifies key features, visualizes distributions, and prepares data for training.

2. **Model Training**:
   - Jupyter Notebook: [Train.ipynb](src/TRAIN.ipynb)
   - Description: This notebook details the training of the predictive model using machine learning techniques. It includes preprocessing steps, model selection (XGBoost classifier in this case), hyperparameter tuning, and evaluation metrics.

3. **Pipeline Execution**:
   - Script: [train.py](src/train.py)
   - Description: Execute this script to run the entire pipeline, including data preprocessing, model training, evaluation, and saving the trained model (`model/1/MY_MODEL.pkl`).

4. **API Deployment**:
   - Script: [app.py](src/app.py)
   - Description: This script launches a FastAPI web server that hosts the trained model. It provides an endpoint (`/predict`) where users can submit JSON data about an NBA player, and the server responds with a prediction whether the player will stay more than 5 years in the league.

### Getting Started with the Pipeline Execution and Running the API

This guide outlines the steps to set up and run the pipeline for the NBA player classifier, including training the model and deploying an API.

#### Prerequisites

1. **Create Conda Environment:**

Make sure that you have conda installed.
```bash
conda create --name MYPROJECT-NAME python=3.11
conda activate MYPROJECT-NAME
```
Replace MYPROJECT-NAME with the desired name for your Conda environment.


2. **Install Poetry:**
```bash
pip install poetry
```

3. **Disable Poetry Virtual Environments :**
```bash
poetry config virtualenvs.create false --local
````

4. **Install Dependencies:**
```bash
poetry install
```

### Running the Pipeline

#### Training the Model

Execute the `train.py` script to run the entire pipeline, including data preprocessing, model training, and evaluation:

```bash
python src/train.py
```

This script will train the NBA player classifier model and save it as `model/1/MY_MODEL.pkl`.

### Running the API

#### Launching the API

Use `uvicorn` to launch the FastAPI server, which provides an endpoint for real-time predictions based on the trained model:

```bash
poetry run uvicorn src.app:app --reload
```