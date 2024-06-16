from typing import Any
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings


class PathsConfig(BaseModel):
    # Preprocess
    RAW_DATA: str = Field(
        default="data/raw/nba_logreg.csv",
        description="Path where to load rawdata.",
    )

    # Train
    TRAIN_DATA: str = Field(
        default="data/train/train.csv",
        description="Path where to load train data.",
    )
    TEST_DATA: str = Field(
        default="data/train/test.csv",
        description="Path where to load test data.",
    )
    TRAIN_MODEL: str = Field(
        default="model/1",
        description="Path where to save and load the model",
    )


class ModelConfig(BaseSettings):
    MODEL_NAME: str = Field(
        default="MYMODEL",
        description="Name of the model to train.",
    )
    NUM_COLS: list[str] = Field(
        default=[
            "GP",
            "MIN",
            "PTS",
            "FGM",
            "FGA",
            "FG%",
            "3P Made",
            "3PA",
            "3P%",
            "FTM",
            "FTA",
            "FT%",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
        ],
        description="List of numeric columns to use for the model.",
    )
    LABEL_COL: str = Field(
        default="TARGET_5Yrs",
        description="Label col",
    )
    DROP_DUP_COLS: str = Field(
        default="Name",
        description="Columns to use when dropping duplicates.",
    )
    GBC_CONFIG: dict[str, Any] = Field(
        default={
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "random_state": 42,
            "scale_pos_weight": 0.6,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        description="Parameters to pass to the model",
    )
    TUNER_NR_TRIALS: int = Field(
        default=20,
        description="Number of trials to run for tuning the model (0 desactivates the tuner)",
    )
    TUNER_PARAMS: dict[str, Any] = Field(
        default={
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
            "model__reg_alpha": [0.1, 0.5, 1.0],
            "model__reg_lambda": [0.1, 0.5, 1.0],
            "model__scale_pos_weight": [1.0],
        },
        description="Parameters to tune the model",
    )


class TrainConfig(BaseSettings):
    EPOCHS: int = Field(
        default=100,
        description="Number of epochs to train the model.",
    )
    TEST_SPLIT_PERCENTAGE: float = Field(
        default=0.2,
        description="Percentage of data to use for testing.",
    )
    TRAIN_TEST_SPLIT_SHUFFLE: bool = Field(
        default=True,
        description="Whether to shuffle the data before splitting into training and testing.",
    )


class Config(BaseSettings):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
