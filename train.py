import json
import os
from dataclasses import asdict

from data import (
    build_concat_train_dataset,
    build_eval_dataset_dict,
    get_compute_metrics_dict,
    get_data_collator,
)
from engine import (
    StopEvaluationAfterOneStepCallback,
    StopTrainingAfterOneStepCallback,
    TrainerWithGenToEval,
)

from models import build_model_and_tokenizer, count_parameters, parse_args


def train():
    args = parse_args()
    args.run_name = args.output_dir.split("/")[-1]
    args.logging_dir = os.path.join(args.output_dir, "runs")
    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
    _ = count_parameters(model, layers=False)

    train_dataset = build_concat_train_dataset(
        tokenizer=tokenizer,
        transform=(
            model.vision_processor if hasattr(model, "vision_processor") else None
        ),
        **asdict(args),
    )
    eval_dataset_dict = build_eval_dataset_dict(
        tokenizer=tokenizer,
        transform=(
            model.vision_processor if hasattr(model, "vision_processor") else None
        ),
        **asdict(args),
    )
    data_collator = get_data_collator(tokenizer=tokenizer, **asdict(args))
    compute_metrics_dict = get_compute_metrics_dict(
        dataset_dict=eval_dataset_dict, tokenizer=tokenizer, **asdict(args)
    )

    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = TrainerWithGenToEval(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        data_collator=data_collator,
        compute_metrics=(
            list(compute_metrics_dict.values())[0]
            if compute_metrics_dict is not None
            else None
        ),
    )
    save_config(args)
    trainer.train()

    trainer.save_model()
    print("Trained model saved...")

    print("Moving to Evaluation...")
    if eval_dataset_dict is not None:
        metrics = {}
        for eval_dataset_name, eval_dataset in eval_dataset_dict.items():
            trainer.compute_metrics = compute_metrics_dict[eval_dataset_name]
            metrics.update(
                trainer.evaluate(
                    eval_dataset=eval_dataset,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
            )
        print(metrics)


def save_config(args):
    os.makedirs(args.logging_dir, exist_ok=True)  # Ensure the directory exists
    config_path = os.path.join(args.logging_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(args), f, indent=4)


if __name__ == "__main__":
    train()
