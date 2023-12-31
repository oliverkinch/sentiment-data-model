"""Training of a transformer-based offensive speech classifier."""

import os
from typing import Dict

import hydra
from datasets import Dataset, DatasetDict, load_metric
from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from .load_data import load_splits
import transformers
import logging


class LoggerLogCallback(transformers.TrainerCallback):
    # https://github.com/huggingface/transformers/issues/4624#issuecomment-946415931
    def __init__(self, logger) -> None:
        super().__init__()
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        control.should_log = False
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            self.logger.info(logs)


logger = logging.getLogger(__name__)
log_callback = LoggerLogCallback(logger=logger)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def train_transformer_model(config: DictConfig) -> AutoModelForSequenceClassification:
    """Training of a transformer-based offensive speech classifier.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        AutoModelForSequenceClassification:
            The trained model.
    """
    # Deal with full determinism
    if config.transformer_model.full_determinism:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "4096:8"

    # Load the data
    data_dict = load_splits(config)
    train_df = data_dict["train"]
    val_df = data_dict["val"]
    test_df = data_dict["test"]

    # Only keep the `text` and `label` columns
    train_df = train_df[["text", "label"]]
    val_df = val_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    # Truncate training split if necessary
    if config.train_split_truncation_length > 0:
        train_df = train_df.sample(n=config.train_split_truncation_length)
        # val_df = val_df.sample(n=config.train_split_truncation_length)
        # test_df = test_df.sample(n=config.train_split_truncation_length)

    # Convert the data to Hugging Face Dataset objects
    train = Dataset.from_pandas(train_df, split="train", preserve_index=False)
    val = Dataset.from_pandas(val_df, split="val", preserve_index=False)
    test = Dataset.from_pandas(test_df, split="test", preserve_index=False)

    # Collect the data into a DatasetDict
    dataset = DatasetDict(train=train, val=val, test=test)

    # Get model config
    model_config = config.transformer_model

    # Create the tokeniser
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
    tokenizer.model_max_length = model_config.context_length

    # Create data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=model_config.padding
    )

    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_id,
        use_auth_token=model_config.use_auth_token,
        cache_dir=".cache",
        from_flax=model_config.from_flax,
        num_labels=2,
    )

    # Tokenise the data
    def tokenise(examples: dict) -> dict:
        doc = examples["text"]
        return tokenizer(doc, truncation=True, padding=True)

    dataset = dataset.map(tokenise)

    # Initialise the metrics
    mcc_metric = load_metric("matthews_correlation")
    f1_metric = load_metric("f1")

    # Create the `compute_metrics` function
    def compute_metrics(predictions_and_labels: EvalPrediction) -> Dict[str, float]:
        """Compute the metrics for the transformer model.

        Args:
            predictions_and_labels (EvalPrediction):
                A tuple of predictions and labels.

        Returns:
            Dict[str, float]:
                The metrics.
        """
        # Extract the predictions
        predictions, labels = predictions_and_labels
        predictions = predictions.argmax(axis=-1)

        # Compute the metrics
        mcc = mcc_metric.compute(predictions=predictions, references=labels)[
            "matthews_correlation"
        ]
        f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]

        return dict(mcc=mcc, f1=f1)

    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=model_config.early_stopping_patience
    )

    # Set up output directory
    model_name = model_config.name
    if config.testing:
        output_dir = f"{config.models.dir}/test_{model_name}"
    else:
        output_dir = f"{config.models.dir}/{model_name}"

    # Create the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=model_config.evaluation_strategy,
        logging_strategy=model_config.logging_strategy,
        save_strategy=model_config.save_strategy,
        eval_steps=model_config.eval_steps,
        logging_steps=model_config.logging_steps,
        save_steps=model_config.save_steps,
        max_steps=model_config.max_steps,
        report_to=model_config.report_to,
        save_total_limit=model_config.save_total_limit,
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size,
        learning_rate=model_config.learning_rate,
        warmup_ratio=model_config.warmup_ratio,
        gradient_accumulation_steps=model_config.gradient_accumulation_steps,
        optim=model_config.optim,
        seed=config.seed,
        full_determinism=model_config.full_determinism,
        lr_scheduler_type=model_config.lr_scheduler_type,
        fp16=model_config.fp16,
        metric_for_best_model="mcc",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback, log_callback],
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    print(trainer.evaluate(dataset["test"]))

    # Save the model
    trainer.save_model()

    # Return the model
    return model


# Used to load data and model in notebook
@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_data_model(config: DictConfig):

    # Load the data
    data_dict = load_splits(config)
    train_df = data_dict["train"]
    val_df = data_dict["val"]
    test_df = data_dict["test"]

    # Only keep the `text` and `label` columns
    train_df = train_df[["text", "label"]]
    val_df = val_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    # Truncate training split if necessary
    if config.train_split_truncation_length > 0:
        train_df = train_df.sample(n=config.train_split_truncation_length)
        # val_df = val_df.sample(n=config.train_split_truncation_length)
        # test_df = test_df.sample(n=config.train_split_truncation_length)

    # Convert the data to Hugging Face Dataset objects
    train = Dataset.from_pandas(train_df, split="train", preserve_index=False)
    val = Dataset.from_pandas(val_df, split="val", preserve_index=False)
    test = Dataset.from_pandas(test_df, split="test", preserve_index=False)

    # Collect the data into a DatasetDict
    dataset = DatasetDict(train=train, val=val, test=test)

    # Get model config
    model_config = config.transformer_model

    # Create the tokeniser
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
    tokenizer.model_max_length = model_config.context_length

    # Create data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=model_config.padding
    )

    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_id,
        use_auth_token=model_config.use_auth_token,
        cache_dir=".cache",
        from_flax=model_config.from_flax,
        num_labels=2,
    )

    # Tokenise the data
    def tokenise(examples: dict) -> dict:
        doc = examples["text"]
        return tokenizer(doc, truncation=True, padding=True)

    dataset = dataset.map(tokenise)

    return dataset, model
