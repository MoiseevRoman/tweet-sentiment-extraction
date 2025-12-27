import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from sentiment_span_extractor.data.dataset import TweetSpanDataset
from sentiment_span_extractor.models.span_module import SpanExtractionModule
from sentiment_span_extractor.utils.dvc_utils import ensure_data

logger = logging.getLogger(__name__)


def infer_main(config: DictConfig):
    logging.basicConfig(level=logging.INFO)

    if config.infer_config.input_csv is None:
        raise ValueError("infer_config.input_csv must be specified")
    if config.infer_config.output_csv is None:
        raise ValueError("infer_config.output_csv must be specified")
    if config.infer_config.checkpoint is None:
        raise ValueError("infer_config.checkpoint must be specified")

    train_cfg = config.train_config
    data_cfg = train_cfg.data_config
    model_cfg = train_cfg.model_config

    data_dir = Path("data")
    ensure_data(data_dir)

    df = pd.read_csv(config.infer_config.input_csv)

    texts = df["text"].tolist()
    sentiments = df["sentiment"].tolist()

    dataset = TweetSpanDataset(
        texts=texts,
        sentiments=sentiments,
        selected_texts=None,
        tokenizer_name=model_cfg.backbone_name,
        max_len=data_cfg.max_len,
    )

    checkpoint_path = f"{config.model_save_path}/{config.infer_config.checkpoint}"
    model = SpanExtractionModule.load_from_checkpoint(
        checkpoint_path,
        backbone_name=model_cfg.backbone_name,
        dropout=model_cfg.dropout,
        use_last_two_layers=model_cfg.use_last_two_layers,
        min_words_for_extraction=config.infer_config.min_words_for_extraction,
    )
    model.eval()

    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    selected_texts = []
    start_indices = []
    end_indices = []

    with torch.no_grad():
        for item in dataset:
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            offsets = item["offsets"].cpu().numpy()

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1)
            end_probs = torch.softmax(end_logits, dim=1)

            idx_start = torch.argmax(start_probs[0]).item()
            idx_end = torch.argmax(end_probs[0]).item()

            text = item["text"]
            sentiment = item["sentiment"]

            selected_text = ""
            for ix in range(idx_start, idx_end + 1):
                if ix < len(offsets):
                    offset_start, offset_end = offsets[ix]
                    selected_text += text[offset_start:offset_end]
                    if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                        selected_text += " "

            min_words = config.infer_config.min_words_for_extraction
            if sentiment == "neutral" or len(text.split()) < min_words:
                selected_text = text

            selected_texts.append(selected_text)
            start_indices.append(idx_start)
            end_indices.append(idx_end)

    output_df = pd.DataFrame({
        "textID": df.get("textID", range(len(df))),
        "selected_text": selected_texts,
        "start": start_indices,
        "end": end_indices,
    })

    output_path = Path(config.infer_config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    infer_main(config)


if __name__ == "__main__":
    main()
