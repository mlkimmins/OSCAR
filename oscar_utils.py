"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import datasets
import random
import pandas as pd

# class OscarUtils(object):
#     def __init__(self):
#         pass
    
def show_random_elements(dataset, num_examples=4):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))



### this function takes in an input and sends it as a 
###
def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    #Returns a dictionary containing the encoded sequence or sequence pair 
    # and additional information: the mask for sequence classification and 
    # the overflowing elements if a max_length is specified.
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = tokenizer(
        batch["abstract"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
    )

    #Hugingface tokenizer returns the attention mask and input ids.
    #     input_ids: list of token ids to be fed to a model
    # token_type_ids: list of token type ids to be fed to a model
    # attention_mask: list of indices specifying which tokens should be attended to by the model
    # overflowing_tokens: list of overflowing tokens sequences if a max length is specified and return_overflowing_tokens=True.
    # special_tokens_mask: if adding special tokens, this is a list of [0, 1], with 0 specifying special added tokens and 1 specifying sequence tokens.
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    # len(batch["input_ids"]) = minibatch size ( or batch_size)
    # len(batch["input_ids"][0]) = max_input_length
    # in effect a 2-d Python List of dimension (batch_size x max_input_length)    
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored. current pad_token_id is = 1 in this model
    # batch labels is the output tokenizer dimension (batch_size x max_output_length)
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

###
### The compute metrics fn takes generated output, pred.predictions and the true label pred.label_ids. these are decoded, then rouge score computed

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }



def generate_answer(batch):
    inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=8192, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    return batch
