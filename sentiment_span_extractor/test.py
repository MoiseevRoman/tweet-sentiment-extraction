import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from sentiment_span_extractor.data.datamodule import TweetSpanDataModule
from sentiment_span_extractor.models.span_module import SpanExtractionModule
from sentiment_span_extractor.utils.dvc_utils import ensure_data

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    logging.basicConfig(level=logging.INFO)

    print("===============================")
    print(f"Model: {config.test_config.checkpoint}")
    print("===============================")

    if config.test_config.checkpoint is None:
        raise ValueError("test_config.checkpoint must be specified")

    train_cfg = config.train_config
    data_cfg = train_cfg.data_config
    model_cfg = train_cfg.model_config

    data_dir = Path("data")
    ensure_data(data_dir)

    test_data_cfg = train_cfg.test_data_config
    datamodule = TweetSpanDataModule(
        train_csv=data_cfg.train_csv,
        test_csv=data_cfg.test_csv,
        tokenizer_name=model_cfg.backbone_name,
        max_len=data_cfg.max_len,
        batch_size=data_cfg.val_batch_size,
        val_batch_size=data_cfg.val_batch_size,
        num_workers=data_cfg.num_workers,
        max_samples=None,
        val_split=test_data_cfg.val_split,
        seed=data_cfg.seed,
    )
    datamodule.setup("test")

    checkpoint_path = f"{config.model_save_path}/{config.test_config.checkpoint}"
    model = SpanExtractionModule.load_from_checkpoint(
        checkpoint_path,
        backbone_name=model_cfg.backbone_name,
        dropout=model_cfg.dropout,
        use_last_two_layers=model_cfg.use_last_two_layers,
        min_words_for_extraction=model_cfg.min_words_for_extraction,
    )

    trainer = pl.Trainer(
        accelerator=train_cfg.accelerator,
        devices=train_cfg.devices,
        precision=train_cfg.precision,
        default_root_dir=test_data_cfg.default_root_dir,
    )

    logger.info("Starting test...")
    results = trainer.test(model, datamodule)
    logger.info(f"Test results: {results}")


if __name__ == "__main__":
    main()
