import json

from model.training_and_validation_pipeline import run_deep_training_and_validation


def main():
    report = run_deep_training_and_validation(save_artifacts=True)
    print(json.dumps(report["random_forest_holdout_evaluation"], indent=2))


if __name__ == "__main__":
    main()
