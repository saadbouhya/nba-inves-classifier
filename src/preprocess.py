from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from config import Config
from loguru import logger


class Preprocess:
    """Preprocessing pipeline."""

    def __init__(self) -> None:
        """Initializes the preprocessing class."""
        self.CONF = Config()

    def init_preprocessor(self) -> None:
        logger.info("Initializing preprocessor.")
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "Imputer",
                    SimpleImputer(strategy="constant", fill_value=0),
                    self.CONF.model.NUM_COLS,
                ),
                (
                    "Scaling",
                    StandardScaler(),
                    self.CONF.model.NUM_COLS,
                ),
            ],
            remainder="drop",
        )
        logger.info("Preprocessor Initialized.")

    def main(self):
        """Main pipeline for the preprocessor initialization."""
        self.init_preprocessor()

        return self.preprocessor
