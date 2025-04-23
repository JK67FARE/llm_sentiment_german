import pandas as pd
from tqdm import tqdm
import argparse
import json
from openai import OpenAI
from pathlib import Path

def prompt_template(x, label):
    return f"""TASK: Sentiment Classification
    INSTRUCTION: Classify the following text into exactly one sentiment category.
    INPUT TEXT: "{x}"

    RULES:
    - You must choose exactly ONE option: {label}
    - Respond with ONLY the chosen word
    - DO NOT add any explanation or additional text
    - DO NOT use punctuation or formatting

    CLASSIFICATION:"""


def create_batch_element(i, text, label):
    batch_element_template = {
        "custom_id": f"{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-4o",
                 "messages": [
                     {"role": "user", "content": prompt_template(text, label)}],
                 "max_tokens": 10}}
    return batch_element_template

def create_batches(INPUT_PATH, OUTPUT_PATH):
    INPUT_PATH = Path(args.input)

    ds = pd.read_feather(INPUT_PATH / "all_ds(1).feather")

    ds = ds[ds.dataset.map(lambda x: x not in ["dataset", "scare", "holiday_check"])]

    scare = pd.read_feather(INPUT_PATH / "scare_reviews_100k.feather")[['text', 'label']]
    scare["dataset"] = "scare"
    hc = pd.read_feather(INPUT_PATH / "holiday_check_100k.feather")[['text', 'label']]
    hc["dataset"] = "holiday_check"

    ds = pd.concat([ds, scare, hc], axis=0)[['text', 'label', 'dataset']]

    ds.reset_index(drop=True, inplace=True)


    labels = ds.groupby("dataset").label.unique().to_dict()
    texts = ds.groupby("dataset").text.unique().to_dict()
    datasets = list(texts.keys())

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    for dataset in datasets:
        tmp_ds = ds[ds["dataset"] == dataset]
        batches = [create_batch_element(f"{i}", text, labels[dataset]) for i, (text, _, dataset) in
                   tqdm(tmp_ds.iterrows())]

        for i, chunk in enumerate(chunker(batches, 50000)):

            with open(OUTPUT_PATH / f'chatgpt-batches-{dataset}_{i}.jsonl', 'w') as outfile:
                for entry in chunk:
                    json.dump(entry, outfile)
                    outfile.write('\n')


def upload_api(api_key, PATH):
    client = OpenAI(api_key=api_key)

    files = list(PATH.glob("*jsonl"))

    files = sorted(files, key=lambda x: str(x).lower())[6:8]

    for file in files:
        ds = file.with_suffix("").name.split("-")[2]
        print(ds)

        batch_input_file = client.files.create(
            file=open(file, "rb"),
            purpose="batch"
        )

        print(batch_input_file)

        batch_input_file_id = batch_input_file.id
        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"{file.with_suffix("").name}"
            }
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='ZeroshotAdding',
                        description='Add Zeroshot Sentiment Polarity Classification to data frame.',
                        )
    parser.add_argument('--input', dest="input", type=str, help="Name of the input file. Required. (feather)")
    parser.add_argument('--openai_key', dest="openai_key", type=str, help="OpenAI API key. Required.")

    args = parser.parse_args()


    TMP_PATH =  Path(args.input).parent / "batches"
    create_batches(args.input, TMP_PATH)
    upload_api(args.openai_key, TMP_PATH)
