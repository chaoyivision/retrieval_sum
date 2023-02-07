from huggingface_hub import notebook_login
#notebook_login()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import datasets
import random
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
import numpy as np

import utils.test as test_util
T5_NAMES =  ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]
PREFIX = "summarize: "
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
INPUT_COL_NAME = "document"
TARGET_COL_NAME = "summary"

# config 
model_checkpoint = "t5-small"
dataset_name = 'xsum'
metric_name = 'rouge'
num_epoch = 4


def preprocess_function(examples):
    inputs = [PREFIX + doc for doc in examples[INPUT_COL_NAME]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[TARGET_COL_NAME], max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

if __name__ == '__main__':


    ### dataset
    raw_datasets = load_dataset(dataset_name)
    print(raw_datasets)
    print(raw_datasets["train"][0])
    #test_util.show_random_elements(raw_datasets["train"])

    ### metric
    metric = load_metric(metric_name)
    print(metric)
    test_util.show_metric(metric)


    ### model & tokenizer & data_collator
    assert model_checkpoint in T5_NAMES, model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #test_util.show_tokenizer(tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    ### process dataset
    print('demo', preprocess_function(raw_datasets['train'][:2]))
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


    ### Fine-tuning the model
    batch_size = 8
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"outputs/{model_name}-finetuned-{dataset_name}-epoch{num_epoch}",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epoch,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


    trainer.train()
    trainer.push_to_hub()

    

