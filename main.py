import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Запуск процессов обучения, тестирования и инференса модели извлечения спанов из твитов"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- TRAIN ---
    train_parser = subparsers.add_parser("train", help="Запуск обучения (train.py)")
    # Аргументы Hydra будут переданы как позиционные аргументы

    # --- TEST ---
    test_parser = subparsers.add_parser("test", help="Запуск тестирования (test.py)")
    test_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к модели", metavar=""
    )

    # --- INFER ---
    infer_parser = subparsers.add_parser("infer", help="Запуск инференса (infer.py)")
    infer_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к модели", metavar=""
    )
    infer_parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Путь к входному CSV файлу",
        metavar="",
    )
    infer_parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Путь к выходному CSV файлу",
        metavar="",
    )

    # --- CONVERT ---
    convert_parser = subparsers.add_parser(
        "convert", help="Запуск конвертации в ONNX (convert.py)"
    )
    convert_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к модели", metavar=""
    )
    convert_parser.add_argument(
        "--output-path",
        type=str,
        required=False,
        default=None,
        help="Путь сохранения ONNX модели",
        metavar="",
    )

    # --- INFER TRITON ---
    infer_triton_parser = subparsers.add_parser(
        "infer-triton", help="Инференс через Triton"
    )
    infer_triton_parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Путь к входному CSV файлу",
        metavar="",
    )
    infer_triton_parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Путь к выходному CSV файлу",
        metavar="",
    )
    infer_triton_parser.add_argument(
        "--triton-url",
        type=str,
        required=False,
        default="http://localhost:8000",
        help="URL Triton сервера",
        metavar="",
    )
    infer_triton_parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="sentiment_span_extractor",
        help="Имя модели в Triton",
        metavar="",
    )

    args, unknown_args = parser.parse_known_args()

    # --- ROUTING ---
    if args.mode == "train":
        cmd = [
            "uv",
            "run",
            "sentiment_span_extractor/train.py",
        ]
        # Передаем все неизвестные аргументы в train.py (это аргументы Hydra)
        cmd.extend(unknown_args)
        subprocess.run(cmd)

    elif args.mode == "test":
        subprocess.run(
            [
                "uv",
                "run",
                "sentiment_span_extractor/test.py",
                f"test_config.checkpoint={args.checkpoint}",
            ]
        )

    elif args.mode == "infer":
        subprocess.run(
            [
                "uv",
                "run",
                "sentiment_span_extractor/infer.py",
                f"infer_config.checkpoint={args.checkpoint}",
                f"infer_config.input_csv={args.input_csv}",
                f"infer_config.output_csv={args.output_csv}",
            ]
        )
    elif args.mode == "convert":
        cmd = [
            "uv",
            "run",
            "sentiment_span_extractor/convert.py",
            f"convert_config.checkpoint={args.checkpoint}",
        ]
        if args.output_path:
            cmd.append(f"convert_config.output_path={args.output_path}")
        subprocess.run(cmd)
    elif args.mode == "infer-triton":
        subprocess.run(
            [
                "uv",
                "run",
                "sentiment_span_extractor/infer_triton.py",
                f"infer_triton_config.input_csv={args.input_csv}",
                f"infer_triton_config.output_csv={args.output_csv}",
                f"infer_triton_config.triton_url={args.triton_url}",
                f"infer_triton_config.model_name={args.model_name}",
            ]
        )


if __name__ == "__main__":
    main()
