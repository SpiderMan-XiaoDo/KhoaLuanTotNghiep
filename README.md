# Xây dựng hệ thống trả lời câu hỏi cho Tiếng Việt

Respository này sẽ xây dựng mô hình phoBERT cho hệ thống trả lời câu hỏi tiếng Việt. Đồng thời respository này sẽ triển khai mô hình PhoBERT đã xây dựng được thông qua việc sử dụng thư viện FastAPI của python

## Tóm tắt
Dự án này được hoàn thành bởi Nguyễn Luôn Mong Đổ, sinh viên ngành KHMT khoa CNTT trường Đại Học Khoa Học Huế.

Trong dự án này, tôi sẽ trình bày cách xây dựng mô hình đồng thời triển khai mô hình ở phía backend bằng thư viện FastAPI của python.

Để xây dựng giao diện cho ứng dụng, hãy tham khảo respository triển khai ứng dụng trả lời câu hỏi ở phía FE tại đường dẫn:


## Quá trình cài đặt, thiết lập môi trường
1. Clone respository:
```
    https://github.com/SpiderMan-XiaoDo/KhoaLuanTotNghiep.git
```
2. Tải mô hình đã được huấn luyện và lưu nó vào thư mục modelv2 tại đường dẫn:  https://www.kaggle.com/datasets/nguyendolikeyou/pretrain-phobert-base/data (Nguyên nhân: pretrain-model có kích thước quá lớn, không thể lưu trữ tại github)

3. Để hiểu hơn về quá trình xây dựng mô hình và cách mà mô hình hoạt động, vui lòng tham khảo notebook tại kaggle với đường dẫn sau:https://www.kaggle.com/code/nguyendolikeyou/fine-tuning-phobertv3
```

```
4. Triển khai API:

```
uvicorn main:app --reload
```
## Tập dữ liệu sử dụng

Quá trình tinh chỉnh mô hình PhoBERT sử dụng tập dữ liệu được mô tả tại đường dẫn: https://paperswithcode.com/dataset/uit-viquad


## Kết quả thu được

Kết quả mô hình thu được với tập Dev-data của tập dữ liệu UIT-dataset:
| Mô hình       | EM      | F1      |
|---------------|---------|---------|
| PhoBERT-base  | 53.8473 | 77.926  |

Kết quả mô hình thu được với tập dữ liệu thu thập được:
<img src = "assets\image\ketqua_valid.png" width = 75%>

## Tài liệu tham khảo
 [1] VLSP 2021 - Vietnamese Machine Reading Comprehension Result
(https://aihub.ml/competitions/public_submissions/35).
 [2] Anh Tuan Nguyen Dat Quoc Nguyen. Phobert: Pre-trained language models for vietnamese. 2020.
 [3]Anh Gia Tuan Nguyen Ngan Luu Thuy Nguyen Kiet Van Nguyen, Duc Vu Nguyen. A vietnamese dataset for evaluating machine reading compre-
hension, 2020.