import pandas as pd
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from pathlib import Path
import argparse


#script for annotating datasets with pretrainend sentiment models via huggingface pipeline. Inputfiles need to be in feather. Usable via argparse 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        prog='sentiment',
                        description='Add Guhr et al. BERT Sentiment Polarity Classification to data frame.',
                        )
    parser.add_argument('--model', dest="model", type=str, help="Name of the pretrained model")
    parser.add_argument('--input', dest="input", type=str, help="Name of the input file. Required. (feather)")
    parser.add_argument('--column', dest="column", default="text", type=str, help="Name of the column containing the "
                                                                                      "input text. If not specified "
                                                                                      "defaults to 'text'")
    args = parser.parse_args()
    input_file = Path(args.input)
    model_name = args.model

    data = pd.read_feather(input_file)

    dataset = Dataset.from_pandas(data)

    pipe = pipeline("text-classification",model=model_name,device = "cuda:0")
       
    results = []
    for out in pipe(KeyDataset(dataset, "text"), batch_size=32,truncation="only_first"):
        results.append(out['label'])
    df = pd.DataFrame(dataset)
    if 'index' in df.columns:
        df.set_index(['index'])
    df[f'{model_name}'] = results
    df[f'{model_name}'] = df[f'{model_name}'].astype(str)
    df = df.reset_index(drop=True)
    df.to_feather(input_file)