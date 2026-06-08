import glob
import json
import os
import pandas as pd
import numpy as np
import argparse
import os.path
import openpyxl

import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
from src.util import _get_model_name_or_path_to_short, _contains_multiple_source_language

parser = argparse.ArgumentParser(description="Export results for multilingual bias bios debiasing")

parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    help="Directory where all persistent data will be stored.",
)

parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    help="Models used to encode the context, e.g. BertModel",
)

parser.add_argument(
    "--results_dir",
    action="store",
    type=str,
    help="Directory where all tpr gap results is stored.",
)

parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    help="The type of bias to mitigate.",
)

parser.add_argument(
    "--target_language",
    action="store",
    type=str,
    help="The language of the evaluation (experiment) corpus"
)


parser.add_argument(
    "--save_path",
    action="store",
    type=str,
    help="the path to save the tex file and spreadsheet",
)

# parser.add_argument(
#     "--summary_xlsx_path",
#     action="store",
#     type=str,
#     help="the path to to speardsheet",
# )

# parser.add_argument(
#     "--sheet_name",
#     action="store",
#     type=str,
#     help="save the experiment result in spreadsheet, the sheetname of the result",
# )

parser.add_argument(
    "--debiasing",
    action="store",
    type=str,
    help="The type of debiasing method used",
)

def _parse_experiment_id(experiment_id):
    experiment=None
    model=None
    model_name_or_path=None
    debiasing=None
    bias_type=None
    seed=None
    source_language=None
    target_language=None
    
    items = experiment_id.split("_")
    # print(f"experiment_id is {experiment_id}, items is {items}")
    
    for item in items:
        id_, val = item[:1], item[2:]

        if id_ == "e":
            experiment = val
        elif id_ == "m":
            model = val
        elif id_ == "c":
            model_name_or_path = val
        elif id_ == "d":
            debiasing = val
        elif id_ == "i":
            bias_type = val
        elif id_ == "s":
            seed = val
        elif id_ == "a":
            source_language = val
        elif id_ == "b":
            target_language = val
        else:
            raise ValueError(f"Unrecognized ID {id_} val {val}")

    return experiment, model, model_name_or_path, debiasing, bias_type, seed, source_language, target_language






# # Mapping function
# def map_source_language_list_to_orderName(source_language):
#     mapping_dict = {
#         'en|pt|ru': 'order1',
#         'en-pt|en-ru|pt-ru': 'order2',
#         'en-pt-ru': 'order3'
#     }
#     simplified_name = []
#     for key, value in mapping_dict.items():
#         if key in source_language:
#             simplified_name.append(value)
#     if not simplified_name:  # If no order mapping is found, return original
#         return source_language
#     return ' + '.join(simplified_name)

# def _pretty_model_name(row):

#     pretty_name_mapping = {
#         "SAL": r"SAL}(",
#         "INLP": r" + \textsc{INLP}",
#     }

#     pretty_order_mapping = {

#     }

#     pretty_name = r" + \textsc{" + pretty_name_mapping[row["debiasing"]] + map_source_language_list_to_orderName(row["source_language"]) + ")"

#     return pretty_name


def _pretty_model_name(row):

    map_source_language_list_to_order = {
        'en': 'en',
        'pt': 'pt',
        'ru': 'ru',
        'en-pt-ru': 'FullyJoint',
        'pt|ru|pt-ru': 'two-subsets-without-target-language',
        'en|ru|en-ru': 'two-subsets-without-target-language',
        'en|pt|en-pt': 'two-subsets-without-target-language',
        'en|pt|ru|en-pt|en-ru|pt-ru': 'two-subsets',
        'en|pt|ru|en-pt|en-ru|pt-ru|en-pt-ru': 'three-subsets'
    }

    if _contains_multiple_source_language(row["source_language"]) and row["debiasing"] == "SAL":
        pretty_name_mapping = {
            "SAL": r"MUSAL}("
        }
    else:
        pretty_name_mapping = {
            "SAL": r"SAL}(",
            "INLP": r" + \textsc{INLP}",
        }

    pretty_name = r" + \textsc{" + pretty_name_mapping[row["debiasing"]] + map_source_language_list_to_order[row["source_language"]]  + ")"

    return pretty_name


# def _pretty_model_name(row):

#     pretty_name_mapping = {
#         "SAL": r"SAL}",
#         "INLP": r" + \textsc{INLP}",
#     }

#     pretty_name = r" + \textsc{" + row["source_language"] + pretty_name_mapping[row["debiasing"]]

#     return pretty_name
    


def _pretty_main_acc(row):
    
    if row['main_task_accuracy_before'] == row['main_task_accuracy_after']:
        return f"{row['main_task_accuracy_after']:.1f}"
    elif row['main_task_accuracy_after'] > row['main_task_accuracy_before']:
        return (
            r"\uag{"
            + f"{(row['main_task_accuracy_after'] - row['main_task_accuracy_before']):.1f}"
            + r"} "
            + f"{row['main_task_accuracy_after']:.1f}"
        )
    else:
        return (
            r"\dab{"
            + f"{(row['main_task_accuracy_before'] - row['main_task_accuracy_after']):.1f}"
            + r"} "
            + f"{row['main_task_accuracy_after']:.1f}"
        )


def _pretty_bias_acc(row):
    
    if row['bias_prediction_accuracy_before'] == row['bias_prediction_accuracy_after']:
        return f"{row['bias_prediction_accuracy_after']:.1f}"
    elif row['bias_prediction_accuracy_after'] > row['bias_prediction_accuracy_before']:
        return (
            r"\ua{"
            + f"{(row['bias_prediction_accuracy_after'] - row['bias_prediction_accuracy_before']):.1f}"
            + r"} "
            + f"{row['bias_prediction_accuracy_after']:.1f}"
        )
    else:
        return (
            r"\da{"
            + f"{(row['bias_prediction_accuracy_before'] - row['bias_prediction_accuracy_after']):.1f}"
            + r"} "
            + f"{row['bias_prediction_accuracy_after']:.1f}"
        )


def _pretty_tpr_gap(row):
    
    if row['tpr_gap_before'] == row['tpr_gap_after']:
        return f"{row['tpr_gap_after']:.1f}"
    elif row['tpr_gap_after'] > row['tpr_gap_before']:
        return (
            r"\ua{"
            + f"{(row['tpr_gap_after'] - row['tpr_gap_before']):.1f}"
            + r"} "
            + f"{row['tpr_gap_after']:.1f}"
        )
    else:
        return (
            r"\da{"
            + f"{(row['tpr_gap_before'] - row['tpr_gap_after']):.1f}"
            + r"} "
            + f"{row['tpr_gap_after']:.1f}"
        )


if __name__ == "__main__":
    args = parser.parse_args()

    model_names = args.model_name_or_path.split(',')
    target_languages = args.target_language.split(',')

    df_all = []
    for df_model_name in model_names:
        for df_target_language in target_languages:
            
            results_dir = f"{args.results_dir}/{df_model_name}/{args.debiasing}/{args.bias_type}/"
            print(f"results_dir is {results_dir}")
            results = []
            for file_path in glob.glob(f"{results_dir}/*.json"):

                with open(file_path, "r") as f:
                    results.extend(json.load(f))

            records = []
                    
            for experiment in results:
                experiment_id = experiment["experiment_id"]
                source_language = 'en'
                target_language = 'en'
                main_task_accuracy_before = round(experiment['main_task_accuracy_before'] * 100, 1)
                main_task_accuracy_after = round(experiment['main_task_accuracy_after'] * 100, 1)
                tpr_gap_before = round(experiment['tpr_gap_before'] * 100, 1)
                tpr_gap_after = round(experiment['tpr_gap_after'] * 100, 1)
                bias_prediction_accuracy_before = round(experiment['bias_prediction_accuracy_before'] * 100, 1)
                bias_prediction_accuracy_after = round(experiment['bias_prediction_accuracy_after'] * 100, 1)

                # experiment, model, model_name_or_path, debiasing, bias_type, seed, source_language, target_language = _parse_experiment_id(experiment["experiment_id"])
                experiment, model, model_name_or_path, debiasing, bias_type, seed, _, _ = _parse_experiment_id(experiment["experiment_id"])
                print(f"bias_type is {bias_type}, args.bias_type is {args.bias_type}")
                records.append(
                    {
                        "experiment_id": experiment_id,
                        "experiment": experiment,
                        "model": model,
                        "debiasing": debiasing,
                        "bias_type": bias_type,
                        "source_language": source_language,
                        "target_language": target_language,
                        "main_task_accuracy_before": main_task_accuracy_before,
                        "main_task_accuracy_after": main_task_accuracy_after,
                        "tpr_gap_before": tpr_gap_before,
                        "tpr_gap_after": tpr_gap_after,
                        "bias_prediction_accuracy_before": bias_prediction_accuracy_before,
                        "bias_prediction_accuracy_after": bias_prediction_accuracy_after,
                    }
                )

            df = pd.DataFrame.from_records(records)
            
            # print(f"df['target_language'] is {df['target_language']},\n\nargs.target_language is {args.target_language}")

            # print(df[df['target_language' == 'EN']])
            # Filter to subset of results.

            # if args.model is not None:
            #     df = df[df["model"] == args.model]
            # if args.source_language is not None:
            #     df = df[df["source_language"] == args.source_language]
            # if args.target_language is not None:
            #     df = df[df["target_language"] == args.target_language]
            # print(f"df_model_name is {df_model_name}, df_target_language is {df_target_language}, bias_type is {args.bias_type}, df is {df}")
            print(f"df['bias_type'] is {df['bias_type']}")

            if args.bias_type is not None:
                df = df[df["bias_type"] == args.bias_type]
            df = df[df["target_language"] == df_target_language]


            # Get pretty model name.
            df["pretty_model_name"] = df.apply(lambda row: _pretty_model_name(row), axis=1)

            # Get main task prediction accuracy in text
            df["pretty_main_acc"] = df.apply(lambda row: _pretty_main_acc(row), axis=1)
            
            # Get gender/race/religion prediction accuracy in text
            df["pretty_bias_acc"] = df.apply(lambda row: _pretty_bias_acc(row), axis=1)

            # Get pretty metric values.
            df["pretty_tpr-gap_value"] = df.apply(lambda row: _pretty_tpr_gap(row), axis=1)


            # # To get proper ordering.
            # df = df.sort_values(by=["debiasing", "assignment"])
            
            df.reset_index(drop = True, inplace = True)


            # Define the custom order for sorting
            order = [
                'en',
                'pt',
                'ru',
                'en-pt-ru',
                'pt|ru|pt-ru',
                'en|ru|en-ru',
                'en|pt|en-pt',
                'en|pt|ru|en-pt|en-ru|pt-ru',
                'en|pt|ru|en-pt|en-ru|pt-ru|en-pt-ru'
            ]

            # Convert the 'source_language' column to a categorical type with the defined order
            df['source_language'] = pd.Categorical(df['source_language'], categories=order, ordered=True)
            # Sort the DataFrame by the 'source_language' column
            df = df.sort_values(by='source_language')
            # Filter out specific language combination
            df = df[df['source_language'] != 'en|pt|ru|en-pt|en-ru|pt-ru']
            

            # "debiased_accuracy": f"{df.iloc[0]['biased_accuracy']:.4f}",
            df_biased_results = pd.DataFrame({'pretty_model_name': r"\textsc{" + f"biased_model" + r"}",
                                'pretty_main_acc': f"{df.iloc[0]['main_task_accuracy_before']:.1f}",
                                'pretty_tpr-gap_value': f"{df.iloc[0]['tpr_gap_before']:.1f}",
                                'pretty_bias_acc': f"{df.iloc[0]['bias_prediction_accuracy_before']:.1f}"},
                                index=[0])

            df = pd.concat([df_biased_results, df])

            df_transposed = df.transpose()
            df_transposed.columns = df_transposed.iloc[0]  # Set the first row as the header
            df_transposed = df_transposed[1:]  # Drop the first row now that it's used as column names
            df_transposed

            # Rows to combine
            acc_row = df_transposed.iloc[0].values  # Row 1 as a NumPy array
            tpr_row = df_transposed.iloc[1].values  # Row 2 as a NumPy array
            # intrinsic_row = df_transposed.iloc[2].values  # Row 3 as a NumPy array

            # Interleave the rows
            # interleaved = np.empty((acc_row.size + tpr_row.size + intrinsic_row.size), dtype=acc_row.dtype)
            interleaved = np.empty((acc_row.size + tpr_row.size), dtype=acc_row.dtype)

            interleaved[0::2] = acc_row  # Fill even indices with acc_row
            interleaved[1::2] = tpr_row  # Fill odd indices with tpr_row
            # interleaved[2::3] = intrinsic_row  # Fill odd indices with tpr_row
            # Create a new DataFrame with original column names (repeated for each row)
            original_columns = df_transposed.columns

            # new_columns = [f"{col}_{suffix}" for col in original_columns for suffix in ['acc', 'tpr-gap', 'intrinsic']]
            new_columns = [f"{col}_{suffix}" for col in original_columns for suffix in ['acc', 'tpr-gap']]

            combined_df = pd.DataFrame([interleaved], columns=new_columns)

            combined_df['model_name'] = f"{df_model_name}"

            map_target_language_to_upper_case = {
                'en': 'EN',
                'pt': 'PT',
                'ru': 'RU'
            }

            combined_df['target_language'] = map_target_language_to_upper_case[df_target_language]
            # put in the front
            combined_df = combined_df[['target_language'] + [col for col in combined_df.columns if col != 'target_language']]

            df_all.append(combined_df)



    
    final_df = pd.concat(df_all, ignore_index=True)
    print(f"final_df is {final_df}")

    final_df = pd.concat(df_all, ignore_index=True)

    # Only keep the columns for the new format
    # target_language, biased_model_acc, biased_model_tpr-gap, +SAL(en)_acc, +SAL(en)_tpr-gap
    # Try to find the correct columns by name or pattern
    keep_columns = []
    for col in final_df.columns:
        if col == 'target_language':
            keep_columns.append(col)
        elif col.endswith('biased_model_acc') or col.endswith('biased_model_tpr-gap'):
            keep_columns.append(col)
        elif '+SAL(en)_acc' in col or '+SAL(en)_tpr-gap' in col:
            keep_columns.append(col)

    # If columns are not named exactly, try to match by partial names
    if not keep_columns or len(keep_columns) < 5:
        # fallback: keep first 5 columns
        keep_columns = list(final_df.columns[:5])

    final_df = final_df[keep_columns]


    os.makedirs(f"{args.save_path}", exist_ok=True)

    with pd.option_context("max_colwidth", 1000):
        with open(
            f"{args.save_path}/tpr-table_i-{args.bias_type}.tex",
            "w",
        ) as f:
            f.write(
                final_df.to_latex(
                    float_format="%.3f",
                    index=False,
                    escape=False,
                    columns=[col for col in final_df.columns if col != 'model_name']
                )
            )

    # Write the new LaTeX table
    with pd.option_context("max_colwidth", 1000):
        with open(
            f"{args.save_path}/tpr-table_i-{args.bias_type}.tex",
            "w",
        ) as f:
            f.write(
                final_df.to_latex(
                    float_format="%.3f",
                    index=False,
                    escape=False,
                    columns=[col for col in final_df.columns if col != 'model_name']
                )
            )
    # Only keep the columns for the new format, with model name as first column
    # model_name, target_language, biased_model_acc, biased_model_tpr-gap, +SAL(en)_acc, +SAL(en)_tpr-gap
    keep_columns = []
    model_col = None
    for col in final_df.columns:
        if col.lower().startswith('model') or 'model_name' in col:
            model_col = col
            break
    if model_col:
        keep_columns.append(model_col)
    for col in final_df.columns:
        if col == 'target_language':
            keep_columns.append(col)
        elif col.endswith('biased_model_acc') or col.endswith('biased_model_tpr-gap'):
            keep_columns.append(col)
        elif '+SAL(en)_acc' in col or '+SAL(en)_tpr-gap' in col:
            keep_columns.append(col)

    # If columns are not named exactly, try to match by partial names
    if not keep_columns or len(keep_columns) < 6:
        # fallback: keep first 6 columns
        keep_columns = list(final_df.columns[:6])

    final_df = final_df[keep_columns]

    # Write the new LaTeX table
    with pd.option_context("max_colwidth", 1000):
        with open(
            f"{args.save_path}/tpr-table_i-{args.bias_type}_newformat.tex",
            "w",
        ) as f:
            f.write(
                final_df.to_latex(
                    float_format="%.3f",
                    index=False,
                    escape=False
                )
            )
