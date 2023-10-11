from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_PATH = "models/dansk_bert/"
MODEL_HF_PATH = "oliverkinch/dansk_bert"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer.model_max_length = 512

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    use_auth_token=False,
    from_flax=False,
    num_labels=2,
    local_files_only=True,
)

model.push_to_hub(MODEL_HF_PATH)
tokenizer.push_to_hub(MODEL_HF_PATH)
