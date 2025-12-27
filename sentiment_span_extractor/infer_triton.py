import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

try:
    import tritonclient.http as httpclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from sentiment_span_extractor.data.dataset import TweetSpanDataset
from sentiment_span_extractor.utils.dvc_utils import ensure_data

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    if not TRITON_AVAILABLE:
        raise ImportError(
            "Triton client not available. Install with: pip install tritonclient[http]"
        )

    logging.basicConfig(level=logging.INFO)

    if config.infer_triton_config.input_csv is None:
        raise ValueError("infer_triton_config.input_csv must be specified")
    if config.infer_triton_config.output_csv is None:
        raise ValueError("infer_triton_config.output_csv must be specified")
    if config.infer_triton_config.triton_url is None:
        raise ValueError("infer_triton_config.triton_url must be specified")
    if config.infer_triton_config.model_name is None:
        raise ValueError("infer_triton_config.model_name must be specified")

    train_cfg = config.train_config
    data_cfg = train_cfg.data_config
    model_cfg = train_cfg.model_config

    data_dir = Path("data")
    ensure_data(data_dir)

    df = pd.read_csv(config.infer_triton_config.input_csv)

    texts = df["text"].tolist()
    sentiments = df["sentiment"].tolist()

    dataset = TweetSpanDataset(
        texts=texts,
        sentiments=sentiments,
        selected_texts=None,
        tokenizer_name=model_cfg.backbone_name,
        max_len=data_cfg.max_len,
    )

    triton_client = httpclient.InferenceServerClient(
        url=config.infer_triton_config.triton_url, verbose=False
    )

    selected_texts = []
    start_indices = []
    end_indices = []

    for item in dataset:
        input_ids = item["input_ids"].unsqueeze(0).numpy()
        attention_mask = item["attention_mask"].unsqueeze(0).numpy()

        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [
            httpclient.InferRequestedOutput("start_logits"),
            httpclient.InferRequestedOutput("end_logits"),
        ]

        response = triton_client.infer(
            config.infer_triton_config.model_name, inputs, outputs=outputs
        )

        start_logits = response.as_numpy("start_logits")
        end_logits = response.as_numpy("end_logits")

        start_probs = np.exp(start_logits) / np.sum(np.exp(start_logits), axis=1, keepdims=True)
        end_probs = np.exp(end_logits) / np.sum(np.exp(end_logits), axis=1, keepdims=True)

        idx_start = np.argmax(start_probs[0])
        idx_end = np.argmax(end_probs[0])

        text = item["text"]
        sentiment = item["sentiment"]
        offsets = item["offsets"].cpu().numpy()

        selected_text = ""
        for ix in range(idx_start, idx_end + 1):
            if ix < len(offsets):
                offset_start, offset_end = offsets[ix]
                selected_text += text[offset_start:offset_end]
                if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                    selected_text += " "

        min_words = config.infer_triton_config.min_words_for_extraction
        if sentiment == "neutral" or len(text.split()) < min_words:
            selected_text = text

        selected_texts.append(selected_text)
        start_indices.append(int(idx_start))
        end_indices.append(int(idx_end))

    output_df = pd.DataFrame({
        "textID": df.get("textID", range(len(df))),
        "selected_text": selected_texts,
        "start": start_indices,
        "end": end_indices,
    })

    output_path = Path(config.infer_triton_config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
