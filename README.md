# EDGAR 8K Earnings Per Share (EPS) Parser

The Python parser to extract the latest EPS from 8K documents in html format. These documents are publicly available on the SEC's EDGAR platform. Extracting this information as promptly as possible can be key in creating an edge on a quantitative trading strategy.

To accomplish this task, parsing steps were build on top of the [sec-parser](https://sec-parser.readthedocs.io/en/latest/) library, which provides an excellent abstraction over SEC filings. Two parsing steps were implemented: (1) extracting relevant tables into a Pandas dataframe, and (2) calculating relevant dollar-prefixed number associations to variations of the phrase "Earnings per Share" using [yiyanghkust/finbert-pretrain](https://huggingface.co/yiyanghkust/finbert-pretrain), a version of the [BERT](https://arxiv.org/pdf/1810.04805) model fine-tuned on 4.9B token financial communication corpus. 

The table parsing step has a 100% accuracy on the sample data set. The word embedding strategy has 22% accuracy on the sample data set, and was done as an experiment.

## Description

An in depth description of the system is in the ProjectWriteup.pdf file.

In summary, the requirements are as follows: 
1. Extract the latest *quarterly* EPS from the filing.
2. When both diluted EPS and basic EPS are present in the filing, prioritize outputting the basic EPS figure.
3. In cases where both adjusted EPS (Non-GAAP) and unadjusted EPS (GAAP) are provided in the filing, opt to output the unadjusted (GAAP) EPS value.
4. If the filing contains multiple instances of EPS data, output the net or total EPS.
5. Notably, enclosed figures in brackets indicate negative values. For instance, in the majority of filings, (4.5) signifies -4.5.
6. In scenarios where the filing lacks an earnings per share (EPS) value but features a loss per share, output values of the loss per share. Output values should always be negative.


## Getting Started

### Dependencies

* This project has been tested on `Python 3.11`. It should work on `>=Python3.9`.
* To run the `word_embeddings` parsing step, it is recommended to have a dedicated GPU. However, it is not necessary.

### Installing

* `git clone https://github.com/rnucuta/trexquant_dataengineer.git`
* `pip install -r requirements.txt`

### Executing program

* To run the vanilla version:
```
python rnucuta_submission.py --input_dir "Training_Filings" --output_file "output_eps.csv"
```
* To run the version that uses word embeddings in the case the table parser can't find any EPS values:
```
python rnucuta_submission.py --input_dir "Training_Filings" --output_file "output_eps.csv" --use_embeddings
```

## Authors

Raymond Nucuta
[email](mailto:rn347@cornell.edu)

## Version History

<!-- * 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]() -->
* 0.1
    * Initial Release with table parsing and preliminary word embeddings steps.

<!-- ## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46) -->