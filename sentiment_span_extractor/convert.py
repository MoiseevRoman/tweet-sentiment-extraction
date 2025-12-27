import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from sentiment_span_extractor.models.span_module import SpanExtractionModule

logger = logging.getLogger(__name__)


def convert_main(config: DictConfig):
    logging.basicConfig(level=logging.INFO)

    if config.convert_config.checkpoint is None:
        raise ValueError("convert_config.checkpoint must be specified")

    train_cfg = config.train_config
    data_cfg = train_cfg.data_config
    model_cfg = train_cfg.model_config

    output_path = config.convert_config.get("output_path", None)
    if output_path is None:
        output_dir = Path("triton/models/sentiment_span_extractor/1")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model.onnx"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = f"{config.model_save_path}/{config.convert_config.checkpoint}"
    logger.info(f"Loading model from {checkpoint_path}")
    model = SpanExtractionModule.load_from_checkpoint(
        checkpoint_path,
        backbone_name=model_cfg.backbone_name,
        dropout=model_cfg.dropout,
        use_last_two_layers=model_cfg.use_last_two_layers,
        min_words_for_extraction=model_cfg.min_words_for_extraction,
    )
    model.eval()

    dummy_input_ids = torch.randint(0, 1000, (1, data_cfg.max_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, data_cfg.max_len), dtype=torch.long)

    logger.info(f"Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["start_logits", "end_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "start_logits": {0: "batch_size", 1: "sequence_length"},
            "end_logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=config.convert_config.opset_version,
    )

    logger.info(f"Model exported successfully to {output_path}")


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    convert_main(config)


if __name__ == "__main__":
    main()
