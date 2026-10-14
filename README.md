# Assignment 1 Starter

Use this package to run the MASSIVE slot-word probing experiment in English and
one Indian language.

## Languages

| Code | Language | MASSIVE locale |
|---|---|---|
| hi | Hindi | hi-IN |
| ur | Urdu | ur-PK |
| bn | Bangla | bn-BD |
| ta | Tamil | ta-IN |
| te | Telugu | te-IN |
| kn | Kannada | kn-IN |
| ml | Malayalam | ml-IN |

The English comparison uses en-US. MASSIVE contains the same request IDs in
every locale, but translations may differ in their number of words and slot
boundaries.

## Setup and run

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python download_massive.py --language en
python download_massive.py --language hi
python prepare_data.py --language en
python prepare_data.py --language hi
python run_probe.py --language hi --output results/hi.json
python validate_results.py results/hi.json
~~~

Replace `hi` with your chosen language code. The first run of `run_probe.py`
also downloads mBERT.

## Protocol

- **Task:** predict SLOT or O for each sampled word.
- **Model:** frozen google-bert/bert-base-multilingual-cased.
- **Data:** matched request-ID pools from the official MASSIVE train,
  validation, and test partitions, followed by balanced word sampling within
  each pool.
- **Runs:** five training samples; the validation and test samples stay fixed.
- **Pooling:** first subword and mean over a word's subwords.
- **Layer choice:** highest mean validation selectivity for each pooling method.
- **Checks:** majority accuracy, a word-type baseline, a control task,
  context-dependent cases, and fertility groups.

`download_massive.py` retrieves Amazon's MASSIVE 1.1 archive from the URL in
`benchmark.yaml`. `prepare_data.py` prints a SHA-256 digest for each prepared
sample.

## Result file

For English and the chosen language, the JSON file contains:

| Field | Contents |
|---|---|
| audit | sample counts, test labels, mean subwords per word, and fertility-group sizes and label balances |
| layer_profile | validation task accuracy, control accuracy, and selectivity at every layer |
| selected_layer | the layer selected separately for each pooling method |
| runs | five test runs with task, majority, word-type, and control accuracy and selectivity |
| against_habit | task and word-type accuracy where the test label differs from the word's usual training label |
| fertility | task accuracy, size, and majority accuracy for one, two, and at least three subwords |
| test_predictions | first-seed predictions, slot types, subword counts, and context-dependent-case flags for optional error analysis |

The control task assigns one random label to each word type. Selectivity is task
accuracy minus control accuracy.

## Submission README

In the README included with your submission, state:

- your Python version and selected language;
- any changes you made to the supplied code;
- any scripts you added for analysis, tables, or plots; and
- whether and how you used a language model or another AI tool.

You do not need to repeat the standard commands printed above if you ran them
unchanged.

## Data and licence

MASSIVE 1.1 is released under CC BY 4.0. The package downloads Amazon's
official archive. Do not submit downloaded data, model files, cached
activations, or your virtual environment.
# Slot-word-probe
