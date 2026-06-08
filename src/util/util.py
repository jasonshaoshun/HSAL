def _is_generative(model):
    # Checks if we are running an autoregressive model.
    return model in [
        "GPT2LMHeadModel",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPGPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel",
        "SelfDebiasGPT2LMHeadModel",
        "SALGPT2LMHeadModel",
        "LlamaModel",
        "SALLlamaModel"
    ]


def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return model in [
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
    ]

def _is_decoder_only(model):
    # Checks if we are running a decoder-only model.
    return model in [
        "LlamaModel",
        "MixtralModel"
    ]

def _contains_multiple_source_language(language):
    return '-' in language


# declare -A model_name_or_path_to_short=(
#     ["meta-llama/Llama-2-7b-hf"]="Llama-2-7b-hf"
#     ["meta-llama/Llama-2-7b-chat-hf"]="Llama2-7b-chat-hf"
#     ["meta-llama/Llama-2-7b-chat"]="Llama2-7b-chat"
#     ["meta-llama/Meta-Llama-3-8B"]="Llama3-8B"
#     ["meta-llama/Meta-Llama-3-8B-Instruct"]="Llama3-8B-Instruct"
#     ["meta-llama/Llama-3.1-8B"]="Llama-3.1-8B"
#     ["meta-llama/Llama-3.2-3B"]="Llama-3.2-3B"
#     ["bert-base-multilingual-uncased"]="mbert-uncased"
#     ["mistralai/Mistral-7B-Instruct-v0.3"]="Mistral-7B-Instruct-v0.3"
#     ["mistralai/Mistral-7B-v0.3"]="Mistral-7B-v0.3"
#     ["mistralai/Mistral-Nemo-Base-2407"]="Mistral-Nemo-Base-2407"
#     ["mistralai/Mistral-Nemo-Instruct-2407"]="Mistral-Nemo-Instruct-2407"
# )

def _get_model_name_or_path_to_short(model_name):
   model_map = {
       "meta-llama/Llama-2-7b-hf": "Llama-2-7b-hf",
       "meta-llama/Llama-2-7b-chat-hf": "Llama2-7b-chat-hf", 
       "meta-llama/Llama-2-7b-chat": "Llama2-7b-chat",
       "meta-llama/Meta-Llama-3-8B": "Llama3-8B",
       "meta-llama/Meta-Llama-3-8B-Instruct": "Llama3-8B-Instruct",
       "meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
       "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
       "bert-base-multilingual-uncased": "Mbert-uncased",
       "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3",
       "mistralai/Mistral-7B-v0.3": "Mistral-7B-v0.3",
       "mistralai/Mistral-Nemo-Base-2407": "Mistral-Nemo-Base-2407",
       "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo-Instruct-2407"
   }
   return model_map.get(model_name, model_name)