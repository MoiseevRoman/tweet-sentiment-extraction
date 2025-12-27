import logging
import subprocess
from pathlib import Path

import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from sentiment_span_extractor.data.datamodule import TweetSpanDataModule
from sentiment_span_extractor.models.span_module import SpanExtractionModule
from sentiment_span_extractor.utils.dvc_utils import ensure_data
from sentiment_span_extractor.utils.logging import init_mlflow, log_hyperparameters
from sentiment_span_extractor.utils.seed import set_seed

logger = logging.getLogger(__name__)


def train_main(config: DictConfig):
    logging.basicConfig(level=logging.INFO)

    train_cfg = config.train_config
    data_cfg = train_cfg.data_config
    model_cfg = train_cfg.model_config

    set_seed(data_cfg.seed)

    data_dir = Path("data")
    ensure_data(data_dir)

    datamodule = TweetSpanDataModule(
        train_csv=data_cfg.train_csv,
        test_csv=data_cfg.test_csv,
        tokenizer_name=model_cfg.backbone_name,
        max_len=data_cfg.max_len,
        batch_size=data_cfg.batch_size,
        val_batch_size=data_cfg.val_batch_size,
        num_workers=data_cfg.num_workers,
        max_samples=data_cfg.max_samples,
        val_split=data_cfg.val_split,
        seed=data_cfg.seed,
    )

    model = SpanExtractionModule(
        backbone_name=model_cfg.backbone_name,
        dropout=model_cfg.dropout,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        use_last_two_layers=model_cfg.use_last_two_layers,
        warmup_steps=train_cfg.scheduler.warmup_steps,
        min_words_for_extraction=model_cfg.min_words_for_extraction,
    )

    mlflow_info = init_mlflow(
        tracking_uri=train_cfg.mlflow_logging.tracking_uri,
        experiment_name=train_cfg.mlflow_logging.experiment_name,
        run_name=train_cfg.mlflow_logging.run_name,
    )

    log_hyperparameters(OmegaConf.to_container(config, resolve=True))

    mlflow_logger = MLFlowLogger(
        experiment_name=train_cfg.mlflow_logging.experiment_name,
        tracking_uri=train_cfg.mlflow_logging.tracking_uri,
        run_id=mlflow_info["run_id"],
    )

    output_dir = Path(config.model_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=config.model_name + "_{val_jaccard:.4f}",
        monitor=train_cfg.training_monitoring,
        mode=train_cfg.training_monitoring_mode,
        save_top_k=train_cfg.save_top_k,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor=train_cfg.training_monitoring,
        mode=train_cfg.training_monitoring_mode,
        patience=train_cfg.early_stopping.patience,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator=train_cfg.accelerator,
        devices=train_cfg.devices,
        precision=train_cfg.precision,
        max_epochs=train_cfg.num_epochs,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
        gradient_clip_val=train_cfg.gradient_clip_val,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=train_cfg.log_every_n_steps,
    )

    # MLflow run уже активен из init_mlflow
    mlflow.log_param("batch_size", data_cfg.batch_size)
    mlflow.log_param("lr", train_cfg.learning_rate)
    mlflow.log_param("num_epochs", train_cfg.num_epochs)
    mlflow.log_param("backbone", model_cfg.backbone_name)

    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        mlflow.log_param("git_commit_hash", commit_hash)
    except Exception:
        pass

    trainer.fit(model, datamodule)

    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
    mlflow.end_run()


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    train_main(config)


if __name__ == "__main__":
    main()
