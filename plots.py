import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_config
import matplotlib.patches as mpatches
from collections import Counter
from agents import Judge
import os

## Plot per category
# Preference Distribution
def plot_preference_dist(df, category, dataset_name):
    # 1. Identify columns ending with "_PREFERENCE"
    preference_cols = [col for col in df.columns if col.endswith("_PREFERENCE")]

    # 2. Collect all unique categories across these preference columns
    all_categories = set()
    for col in preference_cols:
        all_categories.update(df[col].dropna().unique())

    # 3. Create a pastel color palette and map each category to a color.
    palette = sns.color_palette("pastel", n_colors=len(all_categories))
    # Sort categories for consistency.
    sorted_categories = sorted(all_categories, key=lambda x: str(x))
    cat_to_color = dict(zip(sorted_categories, palette))

    # 4. Create a figure with 1 row and 3 columns of subplots.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 5. Plot each pie chart with the column name as the subplot title.
    for i, col in enumerate(preference_cols):
        # Get value counts for each category (including NaN if needed)
        value_counts = df[col].value_counts(dropna=False)
        # Sort categories to maintain consistency.
        categories = sorted(value_counts.index, key=lambda x: str(x))
        slice_colors = [cat_to_color[cat] for cat in categories]
        
        axes[i].pie(
            value_counts.loc[categories],
            autopct='%1.1f%%',
            textprops={'fontsize': 10},
            startangle=140,
            colors=slice_colors,
            labels=None  # Do not display individual labels on the pie slices.
        )
        subtitle = col.split('_')[0].split('/')[1]
        axes[i].set_title(subtitle, fontsize=14)  # Set the subplot title as the column name.
        axes[i].axis('equal')  # Ensures the pie is drawn as a circle.

    # 6. Create a common legend using patches.
    patches = [mpatches.Patch(color=cat_to_color[cat], label=str(cat)) for cat in sorted_categories]
    # Place the legend at the bottom center of the figure.
    fig.legend(handles=patches, loc='lower center', ncol=len(patches), frameon=False, fontsize=14)

    # 7. Add a common title for the entire figure.
    fig.suptitle(f"Preferences Distriution of different LLMs as Judges - {dataset_name} ({category})", fontsize=16)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.savefig(f'results/plots/{dataset_name}_{category}.png')
    plt.show()
    plt.close(fig)

## Plot per Task

def plot_pref_dist_per_task(df, category, dataset_name):
    # Assume df is your DataFrame and it contains a "task" column and some "_PREFERENCE" columns.
    # Identify all columns ending with "_PREFERENCE"
    pref_cols = [col for col in df.columns if col.endswith("_PREFERENCE")]

    # 1. Collect all unique categories across these preference columns (global color mapping)
    all_categories = set()
    for col in pref_cols:
        all_categories.update(df[col].dropna().unique())
    sorted_categories = sorted(all_categories, key=lambda x: str(x))
    palette = sns.color_palette("pastel", n_colors=len(sorted_categories))
    cat_to_color = dict(zip(sorted_categories, palette))

    # 2. Get all unique tasks
    tasks = df['task'].unique()

    # 3. For each task, plot a figure with side-by-side pie charts for each preference column
    for t in tasks:
        task_df = df[df['task'] == t]
        num_plots = len(pref_cols)
        
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        # If there's only one subplot, convert axes to a list for uniform handling.
        if num_plots == 1:
            axes = [axes]
        
        # For each preference column, plot a pie chart for this task.
        for i, col in enumerate(pref_cols):
            # Get counts for each category in this column for the current task.
            value_counts = task_df[col].value_counts(dropna=False)
            # Sort the categories for consistency
            categories = sorted(value_counts.index, key=lambda x: str(x))
            slice_colors = [cat_to_color[cat] for cat in categories]
            
            axes[i].pie(
                value_counts.loc[categories],
                autopct='%1.1f%%',
                startangle=140,
                colors=slice_colors,
                labels=None  # Omit labels on the pie slices
            )
            axes[i].set_title(col)  # Use the preference column name as the subplot title
            axes[i].axis('equal')  # Ensures the pie is drawn as a circle.
        
        # Create a common legend for all subplots in this figure.
        patches = [mpatches.Patch(color=cat_to_color[cat], label=str(cat)) for cat in sorted_categories]
        fig.legend(handles=patches, loc='lower center', ncol=len(patches), frameon=False)
        
        # Add a common title indicating the current task.
        fig.suptitle(f"Preference Distribution for Task: {t}", fontsize=16)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        fig.savefig(f'results/plots/{dataset_name}_{category}_{t}.png')
        plt.show()

## Majority voting
def majority_vote(row, tiebreaking_judge, responder_list):
    def get_tiebreaker_vote(question, responses):
        print('Tie breaker Judge called.')
        preference = tiebreaking_judge.get_preference(question, responses)
        preference = responder_list[int(preference)]
        print('tiebreaker decision:', preference)
        return preference

    pref_cols = [col for col in row.keys() if col.endswith("_PREFERENCE")]
    # Get the votes from all preference columns as a list.
    votes = row[pref_cols].tolist()
    vote_counts = Counter(votes)
    
    # Calculate the threshold: a candidate must receive more than half of the votes.
    required_votes = len(votes) // 2 + 1
    
    # Check if any candidate meets or exceeds the required votes.
    for candidate, count in vote_counts.items():
        if count >= required_votes:
            return candidate
    
    # If no candidate has the required majority, call the tie-breaker.
    print(f'Result was {vote_counts}')
    return get_tiebreaker_vote(row["question"], row[votes])

def plot_majority_vote_with_tiebreaker(df, category, dataset_name):
    # Compute the counts of each preference value.
    value_counts = df["majority_vote"].value_counts(dropna=False)

    # Sort the categories for consistent ordering.
    sorted_categories = sorted(value_counts.index, key=lambda x: str(x))

    # Create a pastel color palette matching the number of unique categories.
    palette = sns.color_palette("pastel", n_colors=len(sorted_categories))
    # Map each category to a specific color.
    cat_to_color = dict(zip(sorted_categories, palette))
    # Create a list of colors corresponding to the sorted categories.
    slice_colors = [cat_to_color[cat] for cat in sorted_categories]

    # Plot the pie chart.
    plt.figure(figsize=(6, 6))
    plt.pie(
        value_counts.loc[sorted_categories],
        labels=None, # Do not display individual labels on the pie slices.
        autopct='%1.1f%%',
        startangle=140,
        colors=slice_colors)
    # 6. Create a common legend using patches.
    patches = [mpatches.Patch(color=cat_to_color[cat], label=str(cat)) for cat in sorted_categories]
    # Place the legend at the bottom center of the figure.
    plt.legend(handles=patches, loc='lower center', ncol=len(patches), frameon=False, fontsize=14)

    plt.title(f"Majority Voting - {category}", fontsize=16)
    plt.axis('equal')  # Draw the pie as a circle.
    plt.tight_layout()
    plt.savefig(f'results/plots/{dataset_name}_majority_voting_{category}.png')
    plt.show()

def get_bias(scores, judges_list):
    # self bias
    bias_df = pd.DataFrame(columns=['self_bias', 'equal', 'self_less'])
    for judge in judges_list:
        # judge = 'judge_'+llm_judge.split('/')[0]
        judgement_cols = [col for col in scores.keys() if col.startswith('judge_'+judge)]
        print('\njudge:', judge)
        print('judgement cols:', judgement_cols)
        self_col = 'judge_'+judge+'_model_'+judge
        print('self:', self_col)
        judgement_cols.remove(self_col)
        print('rest:', judgement_cols)
        # self_bias = (scores[self_col] > scores[judgement_cols]).all(axis=1)
        def is_self_greater(row):
            # If self_col is greater than the maximum of the other columns, then it's greater than all of them.
            return row[self_col] > row[judgement_cols].max()
        def is_self_equal(row):
            # return row[self_col] == row[judgement_cols].max()
            return (row[self_col] == row[judgement_cols]).all()
            # for col in judgement_cols:
            #     if row[self_col] > row[col] or row[self_col] < row[col]:
            #         return False
            # return True
        def is_self_less(row):
            return row[self_col] < row[judgement_cols].max()
        self_bias = scores.apply(is_self_greater, axis=1)
        random_preference = scores.apply(is_self_equal, axis=1)
        self_less = scores.apply(is_self_less, axis=1)
        print('bias count:', self_bias.sum())
        print('equal count:', random_preference.sum())
        print('less count:', self_less.sum())
        bias_df.loc[judge, 'self_bias'] = self_bias.sum()
        bias_df.loc[judge, 'equal'] = random_preference.sum()
        bias_df.loc[judge, 'self_less'] = self_less.sum()