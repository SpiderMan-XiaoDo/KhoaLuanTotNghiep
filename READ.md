# System Question Answering with PhoBERT model
This project will utilize the BERT model for creating a Vietnamese Question answering system.

## Abstract
This project was created by Nguyen Luon Mong Do, a HUSC student.
## Installation

1. Clone this respository:
```
    git clone https://github.com/SpiderMan-XiaoDo/themdautiengviet_keras.git
```

2. To learn more about the model-building process and the trained models, please refer to the following link: https://www.kaggle.com/nguyendolikeyou/seq2seq-model-themdautiengviet
```

```
## Dataset

To train the LSTM model, I used this dataset: https://drive.google.com/drive/folders/11mkQBCUNuKxyLZEyfGe61INU0WY-SrR0


## Result

Model's result with Non-attention:
<img src= "assets\image\non_attention.png" width = 75%>

Model's result with Attention:
<img src = "assets\image\attention.png" width = 75%>

## References
 [1] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine trans lation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473,  2014. Published on 2014/09/01

 [2] Matthew Snover, Bonnie Dorr, Rich Schwartz, Linnea Micciulla, and John Makhoul. A study of translation edit rate with targeted human annotation. https://aclanthology.org/2006.amta-papers.25/, 2006
 [3] Re-evaluating the Role of  BLEU in Machine Translation Research, ChrisCallison Burch, Miles Osborne, Philipp Koehn