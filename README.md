# "Introspective Truthfulness in LMs" - Repository for AI safety camp 2023

# Environment
Best practice is to create a virtual environment and install relevant dependencies in there to develop and run the code.

```
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt -r requirements_precommit.txt
```

## Pre-commit
Install `pre-commit` in your local dev environment.
```
pip install pre-commit
pre-commit install
```
This makes applies various hooks before when you commit code. These hooks check the linting and formatting of your code and ensure that code is formatted properly following `flake8` and `black` standards.

You can read more [here](https://pre-commit.com/).

When you get stuck/annoyed by pre-commit rejecting your commit, you may choose to run `git commit -m "your message" --no-verify` or `-n` to skip the hooks. This is not recommended because it bypasses the linting and can introduce trouble for other devs.


# Data

## Human Eval
Install the library to use the `HumanEval` dataset.

```
cd ./data
git clone https://github.com/openai/human-eval
pip install -e human-eval

```
