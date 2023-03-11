# "Introspective Truthfulness in LMs" - Repository for AI safety camp 2023

# Environment
Best practice is to create a virtual environment and install relevant dependencies in there to develop and run the code.

```
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

# Data

## Human Eval
Install the library to use the `HumanEval` dataset.

```
cd ./data
git clone https://github.com/openai/human-eval
pip install -e human-eval

```
