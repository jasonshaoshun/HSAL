#!/bin/bash
# Variables for running batch jobs.

# Directory where all data and results are written.

persistent_dir='/projects/u6cg/sshao/code/spectral_learning/textual_erasure/'
# persistent_dir='/rds/user/ss3047/hpc-work/code/spectral_learning/textual_erasure/'

# ["SALBertForMaskedLM"]="BertModel"
declare -A debiased_model_to_base_model=(
    ["SALBertForMaskedLM"]="BertModel"
    ["SALLlamaModel"]="LlamaModel"
    ["SALGPT2LMHeadModel"]="GPT2Model"
)

declare -A debiased_model_to_biased_model=(
    ["SALBertForMaskedLM"]="BertForMaskedLM"
    ["SALLlamaModel"]="LlamaModel"
    ["SALGPT2LMHeadModel"]="GPT2LMHeadModel"
)



# ["LlamaModel"]="meta-llama/Meta-Llama-3-8B meta-llama/Llama-3.1-8B meta-llama/Llama-3.2-3B" 
# ["MistralModel"]="mistralai/Mistral-7B-Instruct-v0.3 mistralai/Mistral-7B-v0.3 mistralai/Mistral-Nemo-Base-2407 mistralai/Mistral-Nemo-Instruct-2407"
# ["BertForMaskedLM"]="bert-base-multilingual-uncased"

    # ["LlamaModel"]="meta-llama/Meta-Llama-3-8B meta-llama/Llama-3.2-3B"
    # ["MistralModel"]="mistralai/Mistral-7B-v0.3 mistralai/Mistral-Nemo-Base-2407 mistralai/Mistral-Nemo-Instruct-2407"

    # ["LlamaModel"]="meta-llama/Llama-3.1-8B"
    # ["MistralModel"]="mistralai/Mistral-7B-Instruct-v0.3"
    # ["LlamaModel"]="meta-llama/Meta-Llama-3-8B meta-llama/Llama-3.1-8B meta-llama/Llama-3.2-3B" 
    # ["MistralModel"]="mistralai/Mistral-7B-Instruct-v0.3 mistralai/Mistral-7B-v0.3 mistralai/Mistral-Nemo-Base-2407 mistralai/Mistral-Nemo-Instruct-2407"
    # ["BertModel"]="bert-base-multilingual-uncased"
    # ["BertForMaskedLM"]="bert-base-multilingual-uncased"


model_name_or_path=(
    'bert-base-uncased'
    'meta-llama/Llama-3.1-8B'
    'mistralai/Mistral-7B-v0.3'
)

declare -A model_to_model_name_or_path=(
    ["BertModel"]="bert-base-uncased"
    ["LlamaModel"]="meta-llama/Llama-3.1-8B"
    ["MistralModel"]="mistralai/Mistral-7B-v0.3"
    ["AlbertModel"]="albert-base-v2"
    ["RobertaModel"]="roberta-base"
    ["GPT2Model"]="gpt2"
    ["SALBertForMaskedLM"]="bert-base-multilingual-uncased"
    ["SALLlamaModel"]="meta-llama/Llama-2-7b-chat"
    ["SALGPT2LMHeadModel"]="gpt2"
)


declare -A model_name_or_path_to_short=(
    ["meta-llama/Llama-2-7b-hf"]="Llama-2-7b-hf"
    ["meta-llama/Llama-2-7b-chat-hf"]="Llama2-7b-chat-hf"
    ["meta-llama/Llama-2-7b-chat"]="Llama2-7b-chat"
    ["meta-llama/Meta-Llama-3-8B"]="Llama3-8B"
    ["meta-llama/Meta-Llama-3-8B-Instruct"]="Llama3-8B-Instruct"
    ["meta-llama/Llama-3.1-8B"]="Llama-3.1-8B"
    ["meta-llama/Llama-3.2-3B"]="Llama-3.2-3B"
    ["bert-base-multilingual-uncased"]="Mbert-uncased"
    ["bert-base-uncased"]="Bert-uncased"
    ["mistralai/Mistral-7B-Instruct-v0.3"]="Mistral-7B-Instruct-v0.3"
    ["mistralai/Mistral-7B-v0.3"]="Mistral-7B-v0.3"
    ["mistralai/Mistral-Nemo-Base-2407"]="Mistral-Nemo-Base-2407"
    ["mistralai/Mistral-Nemo-Instruct-2407"]="Mistral-Nemo-Instruct-2407"
)



# model_name_or_path=(
#     'bert-base-uncased'
#     'meta-llama/Meta-Llama-3-8B'
#     'meta-llama/Llama-3.1-8B'
#     'meta-llama/Llama-3.2-3B'
#     'mistralai/Mistral-7B-Instruct-v0.3'
#     'mistralai/Mistral-7B-v0.3'
#     'mistralai/Mistral-Nemo-Base-2407'
#     'mistralai/Mistral-Nemo-Instruct-2407'
# )


# sal_lm_models=(
#     "SALBertForMaskedLM"
#     "SALAlbertForMaskedLM"
#     "SALRobertaForMaskedLM"
#     "SALGPT2LMHeadModel"
# )

# sal_lm_models=(
#     "SALBertForMaskedLM"
# )

# masked_lm_models=(
#     "BertForMaskedLM"
# )
# masked_lm_models=(
#     "BertForMaskedLM"
#     "AlbertForMaskedLM"
#     "RobertaForMaskedLM"
# )

# declare -A model_to_model_name_or_path=(
#     ["BertModel"]="bert-base-uncased"
#     ["AlbertModel"]="albert-base-v2"
#     ["RobertaModel"]="roberta-base"
#     ["GPT2Model"]="gpt2"
#     ["BertForMaskedLM"]="bert-base-uncased"
#     ["AlbertForMaskedLM"]="albert-base-v2"
#     ["RobertaForMaskedLM"]="roberta-base"
#     ["GPT2LMHeadModel"]="gpt2"
#     ["BertForSequenceClassification"]="bert-base-uncased"
#     ["AlbertForSequenceClassification"]="albert-base-v2"
#     ["RobertaForSequenceClassification"]="roberta-base"
#     ["GPT2ForSequenceClassification"]="gpt2"
#     ["SentenceDebiasBertModel"]="bert-base-uncased"
#     ["SentenceDebiasAlbertModel"]="albert-base-v2"
#     ["SentenceDebiasRobertaModel"]="roberta-base"
#     ["SentenceDebiasGPT2Model"]="gpt2"
#     ["SentenceDebiasBertForMaskedLM"]="bert-base-uncased"
#     ["SentenceDebiasAlbertForMaskedLM"]="albert-base-v2"
#     ["SentenceDebiasRobertaForMaskedLM"]="roberta-base"
#     ["SentenceDebiasGPT2LMHeadModel"]="gpt2"
#     ["SentenceDebiasBertForSequenceClassification"]="bert-base-uncased"
#     ["SentenceDebiasAlbertForSequenceClassification"]="albert-base-v2"
#     ["SentenceDebiasRobertaForSequenceClassification"]="roberta-base"
#     ["SentenceDebiasGPT2ForSequenceClassification"]="gpt2"
#     ["INLPBertModel"]="bert-base-uncased"
#     ["INLPAlbertModel"]="albert-base-v2"
#     ["INLPRobertaModel"]="roberta-base"
#     ["INLPGPT2Model"]="gpt2"
#     ["INLPBertForMaskedLM"]="bert-base-uncased"
#     ["INLPAlbertForMaskedLM"]="albert-base-v2"
#     ["INLPRobertaForMaskedLM"]="roberta-base"
#     ["INLPGPT2LMHeadModel"]="gpt2"
#     ["INLPBertForSequenceClassification"]="bert-base-uncased"
#     ["INLPAlbertForSequenceClassification"]="albert-base-v2"
#     ["INLPRobertaForSequenceClassification"]="roberta-base"
#     ["INLPGPT2ForSequenceClassification"]="gpt2"
#     ["CDABertModel"]="bert-base-uncased"
#     ["CDAAlbertModel"]="albert-base-v2"
#     ["CDARobertaModel"]="roberta-base"
#     ["CDAGPT2Model"]="gpt2"
#     ["CDABertForMaskedLM"]="bert-base-uncased"
#     ["CDAAlbertForMaskedLM"]="albert-base-v2"
#     ["CDARobertaForMaskedLM"]="roberta-base"
#     ["CDAGPT2LMHeadModel"]="gpt2"
#     ["CDABertForSequenceClassification"]="bert-base-uncased"
#     ["CDAAlbertForSequenceClassification"]="albert-base-v2"
#     ["CDARobertaForSequenceClassification"]="roberta-base"
#     ["CDAGPT2ForSequenceClassification"]="gpt2"
#     ["DropoutBertModel"]="bert-base-uncased"
#     ["DropoutAlbertModel"]="albert-base-v2"
#     ["DropoutRobertaModel"]="roberta-base"
#     ["DropoutGPT2Model"]="gpt2"
#     ["DropoutBertForMaskedLM"]="bert-base-uncased"
#     ["DropoutAlbertForMaskedLM"]="albert-base-v2"
#     ["DropoutRobertaForMaskedLM"]="roberta-base"
#     ["DropoutGPT2LMHeadModel"]="gpt2"
#     ["DropoutBertForSequenceClassification"]="bert-base-uncased"
#     ["DropoutAlbertForSequenceClassification"]="albert-base-v2"
#     ["DropoutRobertaForSequenceClassification"]="roberta-base"
#     ["DropoutGPT2ForSequenceClassification"]="gpt2"
#     ["SelfDebiasBertForMaskedLM"]="bert-base-uncased"
#     ["SelfDebiasAlbertForMaskedLM"]="albert-base-v2"
#     ["SelfDebiasRobertaForMaskedLM"]="roberta-base"
#     ["SelfDebiasGPT2LMHeadModel"]="gpt2"
#     ["SALBertModel"]="bert-base-uncased"
#     ["SALAlbertModel"]="albert-base-v2"
#     ["SALRobertaModel"]="roberta-base"
#     ["SALGPT2Model"]="gpt2"
#     ["SALBertForMaskedLM"]="bert-base-uncased"
#     ["SALAlbertForMaskedLM"]="albert-base-v2"
#     ["SALRobertaForMaskedLM"]="roberta-base"
#     ["SALGPT2LMHeadModel"]="gpt2"
#     ["SALBertForSequenceClassification"]="bert-base-uncased"
#     ["SALAlbertForSequenceClassification"]="albert-base-v2"
#     ["SALRobertaForSequenceClassification"]="roberta-base"
#     ["SALGPT2ForSequenceClassification"]="gpt2"
# )


# declare -A sal_lm_models_to_sal_models=(
#     ["SALBertForMaskedLM"]="SALBertModel"
#     ["SALAlbertForMaskedLM"]="SALAlbertModel"
#     ["SALRobertaForMaskedLM"]="SALRobertaModel"
#     ["SALGPT2LMHeadModel"]="SALGPT2Model"
# )
