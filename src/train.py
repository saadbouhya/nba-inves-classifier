from config import Config
from loguru import logger
from model import Model
from preprocess import Preprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import joblib


class Train:
    def __init__(self):
        """Training pipeline representing all the functional blocks needed to train the model."""
        self.CONF = Config()

    def load_data(self) -> None:
        """Loading input csv file"""
        logger.info("Loading input data")
        self.ds = pd.read_csv(self.CONF.paths.RAW_DATA)
        self.ds = self.ds.fillna(0)
        self.ds.drop_duplicates(
            subset=self.CONF.model.DROP_DUP_COLS, keep=False, inplace=True
        )
        logger.info("Input data loaded")

    def train_test_split(self) -> None:
        """Splitting the dataset into training and test sets."""
        logger.info("Splitting the dataset.")
        self.ds_train, self.ds_test = train_test_split(
            self.ds,
            test_size=self.CONF.train.TEST_SPLIT_PERCENTAGE,
            stratify=self.ds[self.CONF.model.LABEL_COL],
            random_state=42,
        )
        logger.info("Dataset splitted into train and test.")
        self.ds_train.to_csv(self.CONF.paths.TRAIN_DATA)
        self.ds_test.to_csv(self.CONF.paths.TEST_DATA)
        logger.info("Datasets saved to local directory.")

    def resample_train_data(self) -> None:
        logger.info("Oversampling train set.")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(
            self.ds_train[self.CONF.model.NUM_COLS],
            self.ds_train[self.CONF.model.LABEL_COL],
        )
        self.ds_train = pd.concat([X_res, y_res], axis=1)
        logger.info("Train set oversampled.")

    def init_model(self) -> None:
        """Initializing the model architecture."""
        self.model_instance = Model()
        self.model = self.model_instance.main()

    def init_preprocessor(self) -> None:
        """Initializing the preprocessor."""
        self.preprocess_instance = Preprocess()
        self.preprocessor = self.preprocess_instance.main()

    def init_train_pipe(self) -> None:
        logger.info("Intializing Train Pipe.")
        self.train_pipe = Pipeline(
            [("preprocessor", self.preprocessor), ("model", self.model)]
        )
        logger.info("Train Pipe Initialized.")

    def fit(self) -> None:
        """Fitting the model."""
        logger.info("Starting the grid search.")
        grid_search = GridSearchCV(
            estimator=self.train_pipe,
            param_grid=self.CONF.model.TUNER_PARAMS,
            cv=3,
            scoring="recall",
            verbose=0,
            n_jobs=-1,
            return_train_score=True,
        )
        grid_search.fit(
            self.ds_train, self.ds_train[self.CONF.model.LABEL_COL]
        )
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_}")
        self.train_pipe.set_params(**grid_search.best_params_)
        self.train_pipe.fit(
            self.ds_train, self.ds_train[self.CONF.model.LABEL_COL]
        )
        logger.info("Model trained.")

    def eval_model(self) -> None:
        """Evaluating the model."""
        logger.info("Evaluating the model.")
        y_pred = self.train_pipe.predict(self.ds_test)
        y_true = self.ds_test[self.CONF.model.LABEL_COL]

        y_prob = self.train_pipe.predict_proba(self.ds_test)[:, 1]

        threshold = 0.5
        y_pred_initial = (y_prob >= threshold).astype(int)

        accuracy_initial = accuracy_score(y_true, y_pred_initial)
        precision_initial = precision_score(y_true, y_pred_initial)
        recall_initial = recall_score(y_true, y_pred_initial)
        confusion_mat_initial = confusion_matrix(
            y_true, y_pred_initial, labels=[1, 0]
        )

        logger.info("Metrics before adjusting threshold:")
        logger.info(f"Test Accuracy: {accuracy_initial:.4f}")
        logger.info(f"Test Precision: {precision_initial:.4f}")
        logger.info(f"Test Recall: {recall_initial:.4f}")
        logger.info(f"Confusion Matrix:\n{confusion_mat_initial}")

        new_threshold = 0.35
        y_pred_adjusted = (y_prob >= new_threshold).astype(int)

        accuracy_adjusted = accuracy_score(y_true, y_pred_adjusted)
        precision_adjusted = precision_score(y_true, y_pred_adjusted)
        recall_adjusted = recall_score(y_true, y_pred_adjusted)
        confusion_mat_adjusted = confusion_matrix(
            y_true, y_pred_adjusted, labels=[1, 0]
        )

        logger.info("Metrics after adjusting threshold:")
        logger.info(f"Threshold adjusted to: {new_threshold}")
        logger.info(f"Test Accuracy: {accuracy_adjusted:.4f}")
        logger.info(f"Test Precision: {precision_adjusted:.4f}")
        logger.info(f"Test Recall: {recall_adjusted:.4f}")
        logger.info(f"Confusion Matrix:\n{confusion_mat_adjusted}")

    def save_model(self) -> None:
        """Saves the model to a local directory."""
        logger.info("Saving model.")
        joblib.dump(
            self.train_pipe,
            f"{self.CONF.paths.TRAIN_MODEL}/{self.CONF.model.MODEL_NAME}.pkl",
        )
        logger.info("Model saved.")

    def main(self) -> None:
        """Main pipeline for the model training."""
        logger.info("Starting Train pipeline")
        self.load_data()
        self.train_test_split()
        self.resample_train_data()
        self.init_model()
        self.init_preprocessor()
        self.init_train_pipe()
        self.fit()
        self.eval_model()
        self.save_model()
        logger.info("Train pipeline - ✅ OK!")
