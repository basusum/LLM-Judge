from utils import *
# from experiments import *
import argparse

os.environ["litellm_key"]="sk-EtvD1bxoKo1pMumd_EwIUg"

def run_experiment(config, **kwargs):
    if kwargs['respond']:
        generate_responses(config, start_index=kwargs['start_index'], end_index=kwargs['end_index'])
    if kwargs['score']:
        judge_score(config, start_index=kwargs['start_index'], end_index=kwargs['end_index'])
    if kwargs['prefer']:
        judge_preference(config, start_index=kwargs['start_index'], end_index=kwargs['end_index'])

def main():
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Run experiments for a specific dataset and category."
    )
    parser.add_argument(
        "--dataset", type=str, default='livebench',
        help="The dataset key in config (e.g., 'livebench')."
    )
    parser.add_argument(
        "--category", type=str, default='instruction_following',
        help="The category within the dataset (e.g., 'instruction_following')."
    )
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="The starting index for the experiment."
    )
    parser.add_argument(
        "--end_index", type=int, default=None,
        help="The ending index for the experiment (optional)."
    )
    parser.add_argument(
        "--respond", type=bool, default=False,
        help="To generate responses."
    )
    parser.add_argument(
        "--score", type=bool, default=False,
        help="To score."
    )
    parser.add_argument(
        "--prefer", type=bool, default=False,
        help="To run judge prefernce."
    )
    
    args = parser.parse_args()
    config = load_config()

    # Validate dataset key
    if args.dataset not in config:
        raise ValueError(f"Dataset '{args.dataset}' not found in the config.")

    dataset_config = config[args.dataset]

    # Validate category key within the dataset
    if args.category not in dataset_config:
        raise ValueError(f"Category '{args.category}' not found in dataset '{args.dataset}' config.")

    run_experiment(dataset_config[args.category], **vars(args))


if __name__ == "__main__":
    main()