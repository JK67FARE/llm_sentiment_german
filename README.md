# Dictionary vs Zeroshot

This repository contains information and code for the Computational Humanities Research 2025 article 'From Dictionaries to LLMs – An Evaluation of
Sentiment Analysis Techniques for German Language Data'

## Usage

### Prepare Data
This repository does not provide the datasets, but they can be downloaded here:

Holidaycheck, Filmstarts, PotTs and Sb10k can be downloaded from the [ Guhr et al. repository](https://github.com/oliverguhr/german-sentiment)\
[GermEval2017-Data](http://ltdata1.informatik.uni-hamburg.de/germeval2017/) \
Due to Copyright issues, the SCARE-dataset is not included and needs to be requested from the [authors]( https://www.romanklinger.de/scare/)\
[BBZ-Goldstandard](https://emporion.gswg.info/receive/emporion_mods_00000014). Note that the annotations in the annotations excel-file are aspect-based and need to be aggregated to the polar-sentiment\
[Lessing](https://github.com/lauchblatt/LessingSentimentEmotionCorpus)\
[German Noval Dataset](https://www.informatik.uni-wuerzburg.de/datascience/projects/nlp/kallimachos-concluded/german-novel-dataset/)\
[Amazon Review](https://huggingface.co/datasets/amazon_reviews_multi)\
[SentiLitKrit](https://github.com/dkltimon/SentiLitKrit_19-II)

To use the annotation scripts the datasets must be a feather-file. You can use the prepare_datasets.py for optimal shape 
### Dictionary annotation
In order to use the quantenda-script quanteda_senti.R install the required libraries and use the directory-structure to make it work. 
The needed wordlists are included, but are also freely available: [GPC](http://www.ulliwaltinger.de/sentiment/), [BPW](https://www.uni-giessen.de/de/fbz/fb02/forschung/research-networks/bsfa/textual_analysis/index), [SLK](https://github.com/dkltimon/SentiLitKrit_19-II/tree/master) ,[BAWL-R](https://www.ewi-psy.fu-berlin.de/psychologie/arbeitsbereiche/allgpsy/Download/index.html). Senti-WS is already included in quanteda-sentiment. 


#### Sentiment Prediction 
For the zeroshot prediction use the zero.py or zero_binary.py.\
The prediciton with the guhr pipe can be done with guhr_senti.py

### LLM Experiments

For the experiments with LLMs and pretrained models newer versions of huggingface and torch are needed. we recommend creating an own environment as well in Python 3.12.

Use llm_senti.py per dataset and model like this. Binary flag is needed to adjust label scope. 
```bash
llm_senti.py --file_name "datasets/scare_reviews_100k.feather" --model "mistralai/Ministral-8B-Instruct-2410"  --binary
```
### Pretrained models
Use pretrained_senti.py per dataset and model like this. 
```bash
llm_senti.py --input "datasets/scare_reviews_100k.feather" --model "oliverguhr/german-sentiment-bert" 
```

### Results
all classification-reports can be found in the folder metrics/. It is possible to use the script eval_dict.py to create own classification reports for the dictionaries, finetuned and NLI approaches as well as evaluation.ipynb for the classification reports of the llm experiments.
