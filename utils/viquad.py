import json
import datasets
from underthesea import word_tokenize

logger = datasets.logging.get_logger(__name__)


class ViQuADConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ViQuADConfig, self).__init__(**kwargs)


class ViQuAD(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ViQuADConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
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
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": "/kaggle/working/new_train.json"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": "/kaggle/working/valid.json"}),
        ]

    def _generate_examples(self, filepath):
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context_raw = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    tokens_context = word_tokenize(context_raw)
                    tokens_2 = [token.replace(" ", "_") for token in tokens_context]

                    # Nối các từ lại và thay thế dấu gạch dưới (_) bằng khoảng trắng
                    context = ' '.join(tokens_2)
                    
                    
                    for qa in paragraph["qas"]:
                        if qa["is_impossible"] is False:
                            answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                            raw_answers = [answer["text"] for answer in qa["answers"]]
                            answers = [ ' '.join([token_ans.replace(" ", "_") for token_ans in word_tokenize(answer)]) for answer in raw_answers]
                        else:
                            answer_starts = [0]
                            answers = ""
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": ' '.join([token_ans.replace(" ", "_") for token_ans in word_tokenize(qa["question"])]),
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1