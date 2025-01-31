import argparse
import pandas as pd
import torch
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sentiment Analysis with LLM')
    parser.add_argument('--file_name', type=str, required=True, help='Path to the input Feather file.')
    parser.add_argument('--model',type=str,required=True,help='Model for classification')
    parser.add_argument('--text_column', type=str, default='text', help='Name of the text column in the dataset.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing text.')
    parser.add_argument('--binary', action='store_true', help='Use binary classification (positive or negative)')
    return parser.parse_args()

def initialize_pipeline(model):
    model_id = model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, legacy=False
    )
    return pipeline(
        'text-generation',
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={
            'torch_dtype': torch.bfloat16,
        },
        device_map='auto',
    )

def classify_sentiment(df, classifier, binary, text_column='text', batch_size=16):
    sentiments = []
    df = df.reset_index(drop=True)
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    valid_labels = ['positive', 'negative'] if binary else ['positive', 'negative', 'neutral']
    labels = ' or '.join(valid_labels)

    prompt_template = '''TASK: Sentiment Classification
    INSTRUCTION: Classify the following text into exactly one sentiment category.
    INPUT TEXT: '{text}'
    
    RULES:
    - You must choose exactly ONE option: {labels}
    - Respond with ONLY the chosen word
    - DO NOT add any explanation or additional text
    - DO NOT use punctuation or formatting
    
    CLASSIFICATION:'''

    for i in tqdm(range(0, len(df), batch_size), desc='Processing Batches', total=total_batches):
        batch_texts = df[text_column][i:i+batch_size].tolist()
        prompts = [prompt_template.format(text=text,labels=labels) for text in batch_texts]
        
        try:
            outputs = classifier(
                prompts,
                max_new_tokens=2,
                temperature=0.1, 
                do_sample=True,
                pad_token_id=classifier.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            batch_sentiments = []
            for output in outputs:
                try:
                    sentiment = output[0]['generated_text'].strip().lower()
                    if sentiment not in valid_labels:
                       sentiment = 'error'
                except:
                    sentiment = 'error'
                batch_sentiments.append(sentiment)
                
        except Exception as e:
            print(f'Error processing batch {i//batch_size}: {e}')
            batch_sentiments = ['error'] * len(batch_texts)
        
        sentiments.extend(batch_sentiments)
        

    return sentiments

def main():
    args = parse_arguments()

    try:
        df = pd.read_feather(args.file_name)
    except Exception as e:
        raise ValueError(f'Failed to read the file "{args.file_name}": {e}')

    classifier = initialize_pipeline(args.model)
    print(f'Initialized {args.model}')
    
    df[args.model] = classify_sentiment(
        df=df,
        classifier=classifier,
        text_column=args.text_column,
        batch_size=args.batch_size,
        binary=args.binary
    )

    # Save the results to the output file
    try:
        df.to_feather(args.file_name)
        print(f'Results saved to "{args.file_name}"')
    except Exception as e:
        raise ValueError(f'Failed to save the output to "{args.file_name}": {e}')

if __name__ == '__main__':
    main()
