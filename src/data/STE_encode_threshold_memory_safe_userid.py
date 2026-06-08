"""
Usage:
  encode_bert_states.py [--input_dir=INPUT_DIR] [--output_dir=OUTPUT_DIR] [--model_name_or_path=MODEL_NAME_OR_PATH] [--model=MODEL] [--shuffle=SHUFFLE] [--dataset_name=DATASET_NAME]


Options:
  -h --help                     show this help message and exit
  --input_dir=INPUT_DIR         input files' directory
  --output_dir=OUTPUT_DIR       write down output file
  --model=MODEL                 model name
  --model_name_or_path=MODEL_NAME_OR_PATH  model name or path
  --shuffle=SHUFFLE                     whether to shuffle the characters in AboutMeClean or not
  --dataset_name=DATASET_NAME           dataset name (e.g., mathematics or stackoverflow)

Encoding text with Bert with two methods: average of all words,
 and the cls token as sentence representation.
"""



import numpy as np
from docopt import docopt
import torch
# import transformers
# from transformers import *
import pickle
from tqdm import tqdm
import pdb
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import sys
current_dic = os.getcwd()
sys.path.append(current_dic)
sys.path.append(os.path.join(current_dic, 'src'))
from src.model import models
import csv
import random
import gc
import re
import country_converter as coco
LIMIT = 500000  # limit to first N samples for faster testing; set to a large number for full data

cc = coco.CountryConverter()
GENDER_TO_CODE = {
    "boy": 0,
    "male": 0,
    "girl": 1,
    "female": 1,
    "unknown": 2,
}


def build_global_country_to_code():
    iso3_values = cc.data["ISO3"].dropna().astype(str).str.upper().unique().tolist()
    valid_iso3 = sorted(
        code for code in iso3_values
        if re.fullmatch(r"[A-Z]{3}", code) and code != "UNK"
    )
    country_to_code = {"UNKNOWN": 0}
    next_code = 1
    for code in valid_iso3:
        if code not in country_to_code:
            country_to_code[code] = next_code
            next_code += 1
    return country_to_code


def normalize_gender_series(series):
    return (
        series.fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "unknown"})
    )

def read_data_file(input_file):
    """
    read the data file with a pickle format or CSV with robust error handling
    :param input_file: input path, string
    :return: the file's content as DataFrame
    """
    print(f"Attempting to read file: {input_file}")

    df = pd.read_json(input_file, lines=True)
    print(f"Successfully read file with standard settings. Shape: {df.shape}")
    return df


def encode_text(model, tokenizer, texts, model_name_or_path, max_length=512):
    """
    texts: list/Series of strings (processed one-by-one; batch_size is ignored)
    Returns:
      avg_np: (N, H) mean pooled last-layer embeddings
      cls_np: (N, H) sentence embedding:
              - llama/mistral: last real token
              - bert-like: token[0]
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    name = str(model_name_or_path).lower()
    is_decoder_only = ("llama" in name) or ("mistral" in name)

    all_avg = []
    all_cls = []

    # If it's a pandas Series, make it a python list
    if hasattr(texts, "tolist"):
        texts = texts.tolist()

    with torch.no_grad():
        for text in tqdm(texts):
            if text is None:
                text = ""
            # avoid issues with NaN / non-string
            text = str(text)

            # IMPORTANT: no padding needed for single input -> avoids pad_token error
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            last_hidden_states = outputs["hidden_states"][-1]  # [1, L, H]

            # mean pooling (ignore padding if any)
            mask = attention_mask.unsqueeze(-1)  # [1, L, 1]
            avg_emb = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [1, H]

            # "CLS" vector rule
            if is_decoder_only:
                last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)  # [1]
                cls_emb = last_hidden_states[0, last_idx.item(), :].unsqueeze(0)  # [1, H]
            else:
                cls_emb = last_hidden_states[:, 0, :]  # [1, H]

            all_avg.append(avg_emb.cpu())
            all_cls.append(cls_emb.cpu())

            # free memory ASAP
            del enc, input_ids, attention_mask, outputs, last_hidden_states, mask, avg_emb, cls_emb
            if device.type == "cuda":
                torch.cuda.empty_cache()

    avg_np = torch.cat(all_avg, dim=0).numpy() if all_avg else np.empty((0, 0))
    cls_np = torch.cat(all_cls, dim=0).numpy() if all_cls else np.empty((0, 0))
    return avg_np, cls_np


def filter_empty_entries(df):
    before = len(df)
    # Drop rows with NA in any required column
    filtered = df.dropna(subset=REQUIRED_COLUMNS)
    # Also remove rows where any required column is just empty string or whitespace
    for col in REQUIRED_COLUMNS:
        filtered = filtered[filtered[col].astype(str).str.strip() != '']
    after = len(filtered)
    num_filtered = before - after
    return filtered, num_filtered

def drop_users_with_inconsistent_aboutme(df, user_col="UserId", about_col="AboutMeClean"):
    """Remove ALL rows for any user that has multiple distinct AboutMeClean values.
    This enforces a 1-to-1 mapping: UserId -> AboutMeClean.
    """
    if user_col not in df.columns:
        raise ValueError(f"Missing required column: {user_col}")
    if about_col not in df.columns:
        raise ValueError(f"Missing required column: {about_col}")

    # Treat NaN as a separate value; AboutMeClean should already be non-empty after filtering.
    nunique = df.groupby(user_col)[about_col].nunique(dropna=False)
    bad_users = nunique[nunique > 1].index

    if len(bad_users) == 0:
        return df, 0, 0

    mask_bad = df[user_col].isin(bad_users)
    removed_rows = int(mask_bad.sum())
    removed_users = int(len(bad_users))
    cleaned = df.loc[~mask_bad].copy()
    return cleaned, removed_rows, removed_users


def encode_aboutme_by_userid(model, tokenizer, df, model_name_or_path, user_col="UserId", about_col="AboutMeClean"):
    """Encode AboutMeClean once per unique UserId, then broadcast embeddings back to all rows.

    Returns:
      avg_all: (N, H)
      cls_all: (N, H)
    """
    
    if user_col not in df.columns:
        raise ValueError(f"Missing required column: {user_col}")
    if about_col not in df.columns:
        raise ValueError(f"Missing required column: {about_col}")

    # One AboutMeClean per user (after cleaning, each user should have exactly one).
    user_df = df.drop_duplicates(subset=[user_col])[[user_col, about_col]].copy()

    user_ids = user_df[user_col].tolist()
    about_texts = user_df[about_col].tolist()

    # Encode only unique users
    print(f"Encoding AboutMeClean for {len(user_ids)} unique users...")
    avg_user, cls_user = encode_text(model, tokenizer, about_texts, model_name_or_path)

    # Map each row's user_id -> row in user_ids
    id_to_pos = {uid: i for i, uid in enumerate(user_ids)}
    row_pos = df[user_col].map(id_to_pos).to_numpy()

    avg_all = avg_user[row_pos]
    cls_all = cls_user[row_pos]
    return avg_all, cls_all


if __name__ == '__main__':

    # Filter out entries with any empty required column and count filtered
    # REQUIRED_COLUMNS = ['text(X)', 'aboutme (complex Z)', 'helpfulness(Y)', 'high-reputation(Z)', 'USA(1) or India (0) (Z)']
    REQUIRED_COLUMNS = ['TextClean', 'AboutMeClean', 'AnswerScore', 'Reputation', 'CountryUniversal', 'DisplayName']



    print(f"torch.__version__ : {torch.__version__}")
    print(f"torch.version.cuda : {torch.version.cuda}")
    print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count() : {torch.cuda.device_count()}")
    print(f"torch.cuda.get_device_name(0) : {torch.cuda.get_device_name(0)}")


    arguments = docopt(__doc__)

    input_dir = arguments['--input_dir']
    output_dir = arguments['--output_dir']
    model_name_or_path = arguments['--model_name_or_path']
    model_ = arguments['--model']
    dataset_name = arguments['--dataset_name']
    # batch_size = int(arguments['--batch_size']) if '--batch_size' in arguments else 1

    if dataset_name not in ['mathematics', 'stackoverflow', 'english', 'superuser']:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    in_file = os.path.join(input_dir, "cleaned.jsonl")
    data = read_data_file(in_file)
    if "gender" not in data.columns:
        print("Column 'gender' not found in cleaned.jsonl. Filling with 'UNKNOWN'.")
        data["gender"] = "UNKNOWN"

    data, num_filtered = filter_empty_entries(data)
    print(f"Filtered out {num_filtered} entries with empty required fields in cleaned.jsonl.")
    print(f"Processing cleaned.jsonl with {len(data)} entries after filtering.")
    print(f"Columns in the data: {data.columns.tolist()}")

    # --- Remove users with inconsistent AboutMeClean (same UserId but different AboutMeClean) ---
    if "UserId" in data.columns and "AboutMeClean" in data.columns:
        data, removed_rows, removed_users = drop_users_with_inconsistent_aboutme(
            data, user_col="UserId", about_col="AboutMeClean"
        )
        if removed_users > 0:
            print(f"Removed {removed_rows} rows from {removed_users} users with inconsistent AboutMeClean.")
        print("Data shape after removing inconsistent users:", data.shape)
    else:
        raise ValueError("Expected columns 'UserId' and 'AboutMeClean' to perform user-level AboutMe de-duplication.")
    print("Full data shape:", data.shape)

    data = data.iloc[:LIMIT].copy()
        
    gc.collect()
    
    print(f"New data shape after truncation: {data.shape}")


    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()

    # data = data[:1000]  # for testing purpose, comment it for real use
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # fallback (rare): add a pad token (requires resizing embeddings)
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            

    # if arguments['--shuffle'] == "False":
    #     print("Not shuffling the AboutMeClean entries.")
    #     tokens_aboutme = tokenize(tokenizer, data['AboutMeClean'])
    # elif arguments['--shuffle'] == "True":
    #     # Option 2: random shuffle the order of AboutMeClean entries
    #     print("Shuffling the AboutMeClean entries.")
    #     shuffled_aboutme = list(data['AboutMeClean'])
    #     random.shuffle(shuffled_aboutme)
    #     tokens_aboutme = tokenize(tokenizer, shuffled_aboutme)
    # else:
    #     raise ValueError("Please specify whether to shuffle the AboutMeClean or not by --shuffle True or False")

    # avg_aboutme, cls_aboutme = encode_text(model, tokens_aboutme, model_name_or_path)

    avg_aboutme, cls_aboutme = encode_aboutme_by_userid(model, tokenizer, data, model_name_or_path, user_col="UserId", about_col="AboutMeClean")


    avg_text, cls_text = encode_text(model, tokenizer, data["TextClean"], model_name_or_path)

    avg_name, cls_name = encode_text(model, tokenizer, data["DisplayName"], model_name_or_path)



    print(f"Text avg: {avg_text.shape}, text cls: {cls_text.shape}, aboutme avg: {avg_aboutme.shape}, aboutme cls: {cls_aboutme.shape}, name avg: {avg_name.shape}, name cls: {cls_name.shape}")

    # if dataset_name == 'mathematics':
    #     helpfulness = (data['AnswerScore'] >= 4).astype(int)
    #     reputation = (data['Reputation'] > 295155).astype(int)
    # elif dataset_name == 'stackoverflow':
    #     helpfulness = (data['AnswerScore'] >= 3).astype(int)
    #     reputation = (data['Reputation'] > 55591).astype(int)
    # else:
    #     raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    helpfulness = data['AnswerScore'].astype(int)
    reputation = data['Reputation'].astype(int)
    user_upvotes = data['UserUpVotes'].astype(int)
    user_downvotes = data['UserDownVotes'].astype(int)
    engagement = data['Engagement'].astype(int)
    
    # Map CountryUniversal to global numeric location codes.
    country_series = (
        data['CountryUniversal']
        .fillna('UNKNOWN')
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({'': 'UNKNOWN'})
    )
    country_to_code = build_global_country_to_code()
    unexpected_countries = sorted(set(country_series.unique()) - set(country_to_code.keys()))
    if unexpected_countries:
        raise ValueError(
            f"Unexpected CountryUniversal values for dataset '{dataset_name}': {unexpected_countries}"
        )
    location = country_series.map(country_to_code).astype(int)

    gender_series = normalize_gender_series(data['gender'])
    unexpected_gender_values = sorted(set(gender_series.unique()) - set(GENDER_TO_CODE.keys()))
    if unexpected_gender_values:
        raise ValueError(
            f"Unexpected gender values for dataset '{dataset_name}': {unexpected_gender_values}"
        )
    gender = gender_series.map(GENDER_TO_CODE).astype(int)

    location_label_values = np.array(
        [label for label, _ in sorted(country_to_code.items(), key=lambda item: item[1])]
    )
    gender_label_values = np.array(["male", "female", "unknown"])

    UserId = data['UserId'].astype(str).to_numpy()
    assert avg_text.shape[0] == helpfulness.shape[0] == reputation.shape[0] == location.shape[0] == gender.shape[0] == UserId.shape[0] == avg_name.shape[0], "Mismatched number of samples."

    os.makedirs(output_dir, exist_ok=True)
    np.savez(f"{output_dir}/raw.npz",
        x_text_avg=avg_text, x_text_cls=cls_text,
        x_aboutme_avg=avg_aboutme, x_aboutme_cls=cls_aboutme,
        x_name_avg=avg_name, x_name_cls=cls_name,
        y_helpfulness=helpfulness, y_reputation=reputation, y_location=location, y_gender=gender, y_user_upvotes=user_upvotes, y_user_downvotes=user_downvotes, y_engagement=engagement,
        x_userid=UserId,
        location_label_values=location_label_values,
        gender_label_values=gender_label_values,
    )

    # # Store the features and labels in the corresponding lists
    # if split == 'train':
    #     x_train_text_avg = avg_text
    #     x_train_text_cls = cls_text
    #     x_train_aboutme_avg = avg_aboutme
    #     x_train_aboutme_cls = cls_aboutme
    #     y_train_helpfulness = helpfulness
    #     y_train_reputation = reputation
    #     y_train_location = location
    # elif split == 'dev':
    #     x_dev_text_avg = avg_text
    #     x_dev_text_cls = cls_text
    #     x_dev_aboutme_avg = avg_aboutme
    #     x_dev_aboutme_cls = cls_aboutme
    #     y_dev_helpfulness = helpfulness
    #     y_dev_reputation = reputation
    #     y_dev_location = location
    # elif split == 'test':
    #     x_test_text_avg = avg_text
    #     x_test_text_cls = cls_text
    #     x_test_aboutme_avg = avg_aboutme
    #     x_test_aboutme_cls = cls_aboutme
    #     y_test_helpfulness = helpfulness
    #     y_test_reputation = reputation
    #     y_test_location = location



    # os.makedirs(output_dir, exist_ok=True)


    # np.savez(f"{output_dir}/en.npz",
    #     x_train_text_avg=x_train_text_avg, x_train_text_cls=x_train_text_cls,
    #     x_train_aboutme_avg=x_train_aboutme_avg, x_train_aboutme_cls=x_train_aboutme_cls,
    #     x_dev_text_avg=x_dev_text_avg, x_dev_text_cls=x_dev_text_cls,
    #     x_dev_aboutme_avg=x_dev_aboutme_avg, x_dev_aboutme_cls=x_dev_aboutme_cls,
    #     x_test_text_avg=x_test_text_avg, x_test_text_cls=x_test_text_cls,
    #     x_test_aboutme_avg=x_test_aboutme_avg, x_test_aboutme_cls=x_test_aboutme_cls,
    #     y_train_helpfulness=y_train_helpfulness, y_dev_helpfulness=y_dev_helpfulness, y_test_helpfulness=y_test_helpfulness,
    #     y_train_reputation=y_train_reputation, y_dev_reputation=y_dev_reputation, y_test_reputation=y_test_reputation,
    #     y_train_location=y_train_location, y_dev_location=y_dev_location, y_test_location=y_test_location
    # )

    print("Encoding completed and saved.")
