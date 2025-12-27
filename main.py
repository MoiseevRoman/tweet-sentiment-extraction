import sys

import hydra
from omegaconf import DictConfig

from sentiment_span_extractor.convert import convert_main
from sentiment_span_extractor.infer import infer_main
from sentiment_span_extractor.infer_triton import infer_triton_main
from sentiment_span_extractor.test import test_main

# Импортируем функции main из скриптов (без декоратора @hydra.main)
from sentiment_span_extractor.train import train_main


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    mode = config.get("mode", None)

    if mode is None:
        print("Usage: uv run main.py mode=<train|test|infer|convert|infer-triton> [additional hydra args]")
        print("\nExamples:")
        print("  uv run main.py mode=train train_config.data_config.max_samples=1500")
        print("  uv run main.py mode=test test_config.checkpoint=model.ckpt")
        print("  uv run main.py mode=infer infer_config.checkpoint=model.ckpt infer_config.input_csv=input.csv infer_config.output_csv=output.csv")
        print("  uv run main.py mode=convert convert_config.checkpoint=model.ckpt")
        print("  uv run main.py mode=infer-triton infer_triton_config.input_csv=input.csv infer_triton_config.output_csv=output.csv")
        sys.exit(1)

    if mode == "train":
        train_main(config)
    elif mode == "test":
        test_main(config)
    elif mode == "infer":
        infer_main(config)
    elif mode == "convert":
        convert_main(config)
    elif mode == "infer-triton":
        infer_triton_main(config)
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: train, test, infer, convert, infer-triton")
        sys.exit(1)


if __name__ == "__main__":
    main()
