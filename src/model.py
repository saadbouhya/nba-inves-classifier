from loguru import logger
from config import Config
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score


class Model:
    """Main class containing model architecture (only the architecture, no related fit nor predict !)."""

    def __init__(self):
        """Initializes the model class."""
        self.CONF = Config()

    def init_model(self) -> None:
        """Initialize the model as a classification using GradientBoostedTreesModel."""
        logger.info("Initializing model 🤖")
        self.model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss"
        )
        logger.info("Model Initialized ✅")

    def main(self) -> None:
        """Main pipeline for the model initialization."""
        self.init_model()

        return self.model
