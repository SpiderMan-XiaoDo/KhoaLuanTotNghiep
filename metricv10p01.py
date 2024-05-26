import collections
import json
import numpy as np
from tqdm.auto import tqdm

def compute_metricsv10p01( metric, start_logits, end_logits, features, examples):

    args_n_best = 20
    args_max_answer_length = 200

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : args_n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : args_n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    try:
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue

                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > args_max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)
                    except Exception as e:
#                             print(f"An error occurred: {e}")
                            continue

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"], 'logit_score': best_answer["logit_score"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": "", 'logit_score': None})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    print("predict: ")
    
    for item1, item2 in zip(predicted_answers, theoretical_answers):
        print("predict: ", item1,"theoritical: ", item2)
    return predicted_answers