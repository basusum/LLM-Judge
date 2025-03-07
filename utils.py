import datasets
import pandas as pd
import json
import os
from agents import *

def load_config():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config

def load_dataset(dataset_path, split):
    data = datasets.load_dataset(dataset_path)
    data_df = pd.DataFrame(data[split])
    return data_df

def get_question(turn):
    return turn[0]

def create_output_file(file_path):
    if os.path.exists(file_path):
       df = pd.read_csv(file_path)
       print('Modifying existing file.')
    else:
        df = pd.DataFrame()
    return df

def find_missing(df):
    missing_flag = False
    # blank_cells = pd.DataFrame(columns=['row_idx', 'col_idx', 'col_name'])
    for row_idx in range(df.shape[0]):
        for col_idx, col_name in enumerate(df.columns):
            cell_value = df.iat[row_idx, col_idx]
            # Check if the cell is NaN or an empty string after stripping whitespace
            if pd.isna(cell_value) or (isinstance(cell_value, str) and cell_value.strip() == ""):
                print(f"Empty cell found at row {row_idx}, column {col_idx} (column name: '{col_name}').")
                # blank_cells.loc[len(blank_cells)] = [row_idx, col_idx, col_name]
                missing_flag = True
    return True if missing_flag else False


def generate_responses(config, start_index=None, end_index=None):
    # Data Load
    data_df = load_dataset(config["data_source"], split='test')
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data_df)
        
    # Response Generation
    response_df = create_output_file(config["response_path"])
    response_df['question_id'] = data_df['question_id']
    response_df['question'] = data_df['turns'].apply(get_question)
    response_df['task'] = data_df['task']
    if 'ground_truth' in data_df.keys():
        response_df['ground_truth'] = data_df['ground_truth']
    
    for model in config["llms"]:
        print('Model:', model)
        respondent = Respondent(model)
        for index in range(start_index, end_index):
            print(f'Responding to Question {index}.')
            response = respondent.get_response(response_df.loc[index,'question'])
            response_df.loc[index, model] = response
            response_df.to_csv(config["response_path"], index=False)

def judge_score(config, start_index=None, end_index=None):
    # load responses
    response_df = pd.read_csv(config["response_path"])
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(response_df)
    judgement_df = create_output_file(config["judgement_path"])
    if judgement_df.empty:
        judgement_df = response_df
    for model in config["judges"]:
        print('\nJudge:', model)
        judge = Judge(model)
        for index in range(start_index, end_index):
            print(f'\nScoring Question {index}.')
            for responding_llm in config["llms"]:
                print('Responding LLM:', responding_llm)
                score, reason = judge.get_score(judgement_df.loc[index,'question'], judgement_df.loc[index, responding_llm])
                if score is not None and reason is not None:
                    score_key = model+'_SCORING_'+responding_llm
                    judgement_df.loc[index, score_key] = score
                    reason_key = model+'_SR_'+responding_llm
                    judgement_df.loc[index, reason_key] = reason
                    judgement_df.to_csv(config["judgement_path"], index=False)
                else:
                    print(f"prompt {index} didn't produce any ouput.")

def judge_preference(config, start_index=None, end_index=None):
    # load responses
    response_df = pd.read_csv(config["response_path"])
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(response_df)
    judgement_df = create_output_file(config["judgement_path"])
    if judgement_df.empty:
        judgement_df = response_df
    for model in config["judges"]:
        print('\nJudge:', model)
        judge = Judge(model)
        for index in range(start_index, end_index):
            row = response_df.loc[index]
            print(f'Judging Preference for Question {index}.')
            preference = judge.get_preference(row['question'], row[config["llms"]])
            if preference is not None:
                preference = config["llms"][int(preference)]
                pref_key = model+'_PREFERENCE'
                judgement_df.loc[index, pref_key] = preference
                judgement_df.to_csv(config["judgement_path"], index=False)
            else:
                print(f"prompt {index} didn't produce any ouput.")


# for MTBench and PPE
def read_json():
    pass