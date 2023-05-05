import argparse
import glob
import os
import subprocess

from src.models.openai_model import get_all_model_strings


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="./data/ambi-bench",
        help="Path to dir containing datasets",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./results",
        help="Directory where to store results",
    )

    return parser.parse_args()


def main():
    args = _get_args()

    data_files = glob.glob(os.path.join(args.data_dir, "*.json"))

    models = get_all_model_strings()

    for model in models:
        for data_path in data_files:

            # category prediction
            subprocess.call(
                [
                    "python",
                    "src/evaluators/eval_ambibench_category_prediction.py",
                    "--ambibench_data_path",
                    data_path,
                    "--model",
                    model,
                    "--output_dir",
                    args.output_dir,
                ]
            )
            # with multiple choice options
            subprocess.call(
                [
                    "python",
                    "src/evaluators/eval_ambibench_category_prediction.py",
                    "--ambibench_data_path",
                    data_path,
                    "--model",
                    model,
                    "--output_dir",
                    args.output_dir,
                    "--use_multiple_choice",
                ]
            )

            # completion
            # subprocess.call(
            #     ["python",
            #      "src/evaluators/eval_ambibench_completion.py",
            #      "--ambibench_data_path", data_path,
            #      "--model", model,
            #      "--output_dir", args.output_dir,
            #      ]
            # )


if __name__ == "__main__":
    main()
