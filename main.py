
import numpy as np
# python -m uvicorn main:app --reload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from metric import *
from datasets import load_dataset
from transformers.models.bartpho.tokenization_bartpho_fast import BartphoTokenizerFast
from transformers import AutoModelForQuestionAnswering, default_data_collator, get_scheduler
from transformers import AutoTokenizer
from torch import nn
import evaluate
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import collections
from metricv10p01 import *


import os
import json
from sklearn.model_selection import train_test_split
from datasets import load_dataset

import random
import string

args_metric = 'squad'
max_length = 256
stride = 128
args_batch_size = 10
device = 'cpu'
args_pretrained_model = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(args_pretrained_model)

args_output_dir = 'D:/WorkSpace/KhoaLuanTotNghiep/modelv2'


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể thay thế "*" bằng danh sách các nguồn cho phép cụ thể
    allow_credentials=True,
    allow_methods=["*"],  # Có thể thay thế "*" bằng danh sách các phương thức HTTP được cho phép
    allow_headers=["*"],  # Có thể thay thế "*" bằng danh sách các tiêu đề HTTP được cho phép
)



tokenizer = BartphoTokenizerFast.from_pretrained("vinai/bartpho-syllable")
tokenizer.is_fast


args_metric = 'squad'
max_length = 256
stride = 128
args_batch_size = 10
device = 'cpu'
args_pretrained_model = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(args_pretrained_model)

args_output_dir = 'D:/WorkSpace/KhoaLuanTotNghiep/modelv2'

args_device = 'cpu'
device = torch.device(args_device)
trained_model = AutoModelForQuestionAnswering.from_pretrained(args_output_dir)
trained_model.to(device)



def generate_random_folder_name(length=8):
    characters = string.ascii_letters + string.digits
    folder_name = ''.join(random.choice(characters) for _ in range(length))
    return folder_name


def preprocess_training_dataset(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
         max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
def preprocess_validation_dataset(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def save_data(data, type):
    # save your preprocessed data
    with open(os.path.join("", type + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent= 4)
    return

def answer_question(context, question, trained_model):
    with open("D:/WorkSpace/KhoaLuanTotNghiep/context_file/train_qa_vi_mailong.json", "r", encoding="utf-8") as f:
        test_data_vf = json.load(f)
    f.close()
    test_data_vf['data'][0]["paragraphs"][0]['qas'][0]["question"] = question
    test_data_vf['data'][0]["paragraphs"][0]['context']= context



    file_content = '''

# Lưu dữ liệu vào file dataloader với tên file được tạo từ fiel dataloader_name.txt
import json
import datasets

logger = datasets.logging.get_logger(__name__)

with open("D:/WorkSpace/Data_Storage_KhoaLuan/dataloader_name.txt", "r") as file:
    temp_file_path = file.read()
print(f"Đã lưu tên thư mục vào file dataloader_name.txt", temp_file_path)
print("Hello")
print("Broro")

class ViQuADConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ViQuADConfig, self).__init__(**kwargs)

class ViQuAD(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ViQuADConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text"
        ),
    ]
    
        

    def _info(self):
        return datasets.DatasetInfo(
            description="UIT-ViQuAD2.0",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"D:/WorkSpace/Data_Storage_KhoaLuan/{temp_file_path}.json"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"D:/WorkSpace/Data_Storage_KhoaLuan/{temp_file_path}.json"}),
        ]

    def _generate_examples(self, filepath):
        logger.info("generating examples from = %s", filepath)
        print(filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        if qa["is_impossible"] is False:
                            answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                            answers = [answer["text"] for answer in qa["answers"]]
                        else:
                            answer_starts = [0]
                            answers = ""
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1
'''

    print(file_content)



    # Tạo tên thư mục ngẫu nhiên
    random_folder_name = generate_random_folder_name()


    # Lưu giá trị tên thư mục vào file dataloader_name.txt
    with open("D:/WorkSpace/Data_Storage_KhoaLuan/dataloader_name.txt", "w") as file:
        file.write(random_folder_name)
        
    with open("D:/WorkSpace/Data_Storage_KhoaLuan/dataloader_name.txt", "r") as file:
        content = file.read()
        
    print(content)

    with open(f"D:/WorkSpace/Data_Storage_KhoaLuan/{content}.py", "w", encoding="utf-8") as file:
        file.write(file_content)

    print(test_data_vf)
    save_data(test_data_vf, f"D:/WorkSpace/Data_Storage_KhoaLuan/{content}")





    mailong_raw_datasets_vf = load_dataset(f"D:/WorkSpace/Data_Storage_KhoaLuan/{content}.py")

    mailong_raw_datasets_vf["train"] = mailong_raw_datasets_vf["train"].filter(lambda x: len(x["answers"]["text"]) == 1)
    mailong_raw_datasets_vf["validation"] = mailong_raw_datasets_vf["validation"].filter(lambda x: len(x["answers"]["text"]) == 1)

    
    mailong_validation_dataset_vf = mailong_raw_datasets_vf["train"].map(
        preprocess_validation_dataset,
        batched=True,
        remove_columns= mailong_raw_datasets_vf["train"].column_names,
    )
   
    print( mailong_raw_datasets_vf["train"]["context"])

    metric = evaluate.load(args_metric)

    mailong_validation_set_vf = mailong_validation_dataset_vf.remove_columns(["example_id", "offset_mapping"])
    mailong_validation_set_vf.set_format("torch")

    mailong_eval_dataloader_vf = DataLoader(
        mailong_validation_set_vf,
        collate_fn=default_data_collator,
        batch_size=args_batch_size
    )

        
    trained_model.eval()
    start_logits = []
    end_logits = []
    print("Evaluation!")
    for batch in tqdm(mailong_eval_dataloader_vf):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = trained_model(**batch)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(mailong_validation_dataset_vf)]
    end_logits = end_logits[: len(mailong_validation_dataset_vf)]

    metrics = compute_metricsv10p01(
        metric, start_logits, end_logits, mailong_validation_dataset_vf, mailong_raw_datasets_vf["train"]
    )
    file_json = f"D:/WorkSpace/Data_Storage_KhoaLuan/{content}.json" # Đường dẫn đến tệp dữ liệu bạn muốn xóa

    # Kiểm tra xem tệp tồn tại hay không trước khi xóa
    if os.path.exists(file_json):
        os.remove(file_json)
        print(f"Tệp '{file_json}' đã được xóa thành công.")
    else:
        print(f"Tệp '{file_json}' không tồn tại.")
    file_squad = f"D:/WorkSpace/Data_Storage_KhoaLuan/{content}.py"
    if os.path.exists(file_squad):
        os.remove(file_squad)
        print(f"Tệp '{file_squad}' đã được xóa thành công.")
    else:
        print(f"Tệp '{file_squad}' không tồn tại.")
    return metrics

@app.get("/answering")
def get_answer(context: str, question: str):
    # try:
        system_answer = answer_question(context, question, trained_model)

        # return system_answer
        print(system_answer)
        return {
            'text': system_answer[0]["prediction_text"],
            'logit_score': str(system_answer[0]["logit_score"])
        }
    # excep
    
    
@app.get("/")
def root_func():
    return {
        "Hello_world" : "hello python"
}