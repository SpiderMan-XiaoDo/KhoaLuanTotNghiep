import numpy as np

import torch
import datasets

from metric import *
from datasets import load_dataset
from transformers.models.bartpho.tokenization_bartpho_fast import BartphoTokenizerFast
from transformers import AutoModelForQuestionAnswering, default_data_collator, get_scheduler
from transformers import AutoTokenizer
from torch import nn
import evaluate
import numpy as np
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

args_metric = 'squad'
max_length = 256
stride = 128
args_batch_size = 10
device = 'cpu'
args_pretrained_model = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(args_pretrained_model)

args_output_dir = 'D:/WorkSpace/KhoaLuanTotNghiep/modelv2'

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

with open("D:/WorkSpace/KhoaLuanTotNghiep/context_file/train_qa_vi_mailong.json", "r", encoding="utf-8") as f:
    test_data_vf = json.load(f)
f.close()



test_data_vf['data'][0]["paragraphs"][0]['qas'][0]["question"] = "Số lượng sách xuất bản của Sherlock Holmes là bao nhiêu"
test_data_vf['data'][0]["paragraphs"][0]['context']= '''Sherlock Holmes (/ˈʃɜːlɒk ˈhoʊmz/) là một nhân vật thám tử tư hư cấu, do nhà văn người Anh Arthur Conan Doyle sáng tạo nên. Tự coi mình là "thám tử tư vấn" trong các câu chuyện, Holmes nổi danh với khả năng quan sát, diễn dịch, khoa học pháp y điêu luyện và suy luận logic tuyệt vời, những yếu tố mà anh áp dụng khi điều tra các vụ án của nhiều dạng khách hàng, bao gồm cả Scotland Yard.

Xuất hiện lần đầu trong tác phẩm Cuộc điều tra màu đỏ năm 1887, nhân vật này dần trở nên nổi tiếng với loạt truyện ngắn đầu tiên trên The Strand Magazine, bắt đầu bằng "Vụ tai tiếng xứ Bohemia" năm 1891. Kể từ đó, những câu chuyện mới lần lượt ra đời cho đến năm 1927. Tổng cộng đã có 4 tiểu thuyết, 56 truyện ngắn và 2 truyện cực ngắn được xuất bản. Các câu chuyện hầu hết lấy bối cảnh vào giữa những năm 1880 và 1914, chỉ trừ một số diễn ra ở thời đại Victoria hoặc Edward. Chúng đa phần được thuật lại qua lời của bác sĩ John H. Watson, một cây viết tiểu sử và là người bạn thân của Holmes. Watson thường song hành cùng Holmes trong các cuộc điều tra và cũng thường chia sẻ với Holmes căn hộ số 221B, phố Baker, Luân Đôn, nơi khởi nguồn của nhiều chuyến phiêu lưu.

Mặc dù không phải là nhân vật thám tử hư cấu đầu tiên, Sherlock Holmes vẫn được xem là nhân vật nổi tiếng nhất.[1] Đến những năm 1990, đã có hơn 25.000 tác phẩm chuyển thể sân khấu, phim, chương trình truyền hình và ấn phẩm có tên vị thám tử này.[2] Sách kỷ lục Guinness liệt kê Holmes là nhân vật văn học được khắc họa nhiều nhất trong lịch sử điện ảnh và truyền hình.[3] Sự phổ biến và danh tiếng của Holmes khiến nhiều người tưởng rằng anh là một nhân vật có thật chứ không phải hư cấu.[4][5] Đây cũng là tiền đề cho sự thành lập của nhiều nhóm văn học hay hội mộ điệu. Những độc giả say mê các câu chuyện về Holmes chính là những người tạo ra thông lệ hiện đại cho khái niệm cộng đồng người hâm mộ.[6] Holmes với những chuyến hành trình của anh đã có ảnh hưởng sâu sắc và lâu dài đến nền văn học bí ẩn cũng như văn hóa đại chúng nói chung, khi những tác phẩm gốc của Conan Doyle hay hàng ngàn câu chuyện được viết bởi các tác giả khác, được chuyển thể thành kịch sân khấu, truyền thanh, truyền hình, phim ảnh, trò chơi điện tử cùng nhiều loại hình truyền thông khác, trong suốt hơn một trăm năm

Cảm hứng sáng tạo nhân vật

Arthur Conan Doyle (1859–1930), người sáng tạo Sherlock Holmes, năm 1914
C. Auguste Dupin của Edgar Allan Poe thường được công nhận là thám tử đầu tiên xuất hiện trên tiểu thuyết và là nguyên mẫu cho nhiều nhân vật sau này bao gồm cả Sherlock Holmes.[7] Conan Doyle đã từng viết, "Mỗi truyện [trinh thám của Poe] đều là gốc rễ cho cả một nền văn học phát triển... Thử hỏi, truyện trinh thám đã ở đâu cho đến khi Poe thổi hơi thở cuộc sống vào nó?"[8] Tương tự, những câu chuyện về Monsieur Lecoq của Émile Gaboriau cũng cực kỳ nổi tiếng vào thời điểm Conan Doyle bắt đầu viết Sherlock Holmes, cách đối thoại và lối cư xử của Holmes đôi khi được phỏng theo Lecoq.[9] Holmes và Watson từng thảo luận về Dupin và Lecoq ngay gần phần đầu của Cuộc điều tra màu đỏ.[10]

Conan Doyle nhắc lại nhiều lần rằng Joseph Bell, một bác sĩ phẫu thuật tại Bệnh viện Hoàng gia Edinburgh, là nguồn cảm hứng có thật cho nhân vật Sherlock Holmes. Doyle gặp Bell vào năm 1877 và từng làm thư ký cho ông này. Cũng như Holmes, Bell nổi tiếng với khả năng đưa ra các đánh giá khái quát từ những quan sát nhỏ.[11] Tuy nhiên, trong bức thư Joseph Bell gửi cho Conan Dolye lại có đoạn: "Bạn mới thật sự là Sherlock Holmes, bản thân bạn biết điều đó mà."[12] Sir Henry Littlejohn, giáo sư khoa luật y tế tại Viện y Đại học Edinburgh, cũng được coi là nguồn cảm hứng cho Holmes. Littlejohn, một bác sĩ pháp y kiêm cán bộ y tế ở Edinburgh, đã cung cấp cho Conan Doyle mối liên hệ giữa nghiên cứu y khoa và điều tra tội phạm.[13]

Cũng có một vài nguồn cảm hứng khả dĩ khác được đề xuất mặc dù chưa bao giờ được Doyle thừa nhận, chẳng hạn như tác phẩm Maximillien Heller của tác giả người Pháp Henry Cauvain. Trong cuốn tiểu thuyết năm 1871 của mình (mười sáu năm trước khi Sherlock Holmes xuất hiện lần đầu tiên), Henry Cauvain đã khắc họa một thám tử đa nhân cách, rầu rĩ, chống đối xã hội, hút thuốc phiện và đang hoạt động ở Paris.[14][15] Dù Conan Doyle thông thạo tiếng Pháp nhưng không rõ liệu ông có từng đọc Maximillien Heller hay chưa.[16] Tương tự, Michael Harrison cho rằng "thám tử tư vấn" tự phong người Đức Walter Scherer có thể là hình mẫu cho Holmes.[17]

Tiểu sử nhân vật hư cấu
Gia đình và giai đoạn đầu đời

Trang bìa phiên bản năm 1887 của Beeton's Christmas Annual, trong đó có sự xuất hiện lần đầu của Holmes (Cuộc điều tra màu đỏ)
Trong tác phẩm của Conan Doyle, thông tin chi tiết về cuộc đời của Sherlock Holmes rất hiếm hoi và thường không rõ ràng. Tuy nhiên, những đề cập ít ỏi về thời niên thiếu và gia đình của Holmes vẫn đủ sức vẽ nên một bức tranh tiểu sử lỏng lẻo cho vị thám tử.

Truyện ngắn "Cung đàn sau cuối" khẳng định Holmes sinh năm 1854 khi miêu tả anh ở tuổi 60 với bối cảnh tháng 8 năm 1914.[18] Cha mẹ của Holmes không được nhắc tới mặc dù anh từng nói rằng "tổ tiên" của mình là "những điền chủ nông thôn". Trong "Người thông ngôn Hy Lạp", Holmes cho biết bà của mình là em gái họa sĩ người Pháp Vernet mà không nói rõ đó là Claude Joseph, Carle hay Horace Vernet. Holmes có một người anh trai là Mycroft, hơn anh 7 tuổi và là một quan chức chính phủ. Mycroft giữ một vị trí công vụ đặc biệt như một dạng cơ sở dữ liệu sống liên quan tới tất cả khía cạnh trong các quyết sách của nhà nước. Holmes mô tả Mycroft là người thông minh hơn trong hai anh em nhưng lưu ý rằng Mycroft không quan tâm đến việc điều tra thực địa và thích dành thời gian ở Câu lạc bộ Diogenes.[19][20]

Holmes nói rằng lần đầu tiên anh bắt đầu phát triển các phương pháp suy luận là khi đang còn học đại học; những vụ án đầu đời mà anh theo đuổi trong vai trò một thám tử nghiệp dư cũng đến từ các sinh viên cùng trường.[21] Holmes quyết định chọn thám tử làm một nghề chuyên nghiệp sau lần gặp gỡ cha của một người bạn.[22]

Cuộc sống với Watson
Holmes (in deerstalker hat) talking to Watson (in a bowler hat) in a railway compartment
Holmes và Watson trong một bức minh họa của Sidney Paget cho "Ngọn lửa bạc"
Khó khăn tài chính khiến Holmes và bác sĩ Watson phải cùng nhau chia sẻ căn hộ tầng trên số 221B, phố Baker, London,[23] thuê của bà Hudson.[24] Holmes làm thám tử trong 23 năm với 17 năm được Watson hỗ trợ.[25] Hầu hết các câu chuyện đều là kiểu truyện khung, được viết theo quan điểm của Watson và là bản tóm tắt những vụ án thú vị nhất của vị thám tử. Holmes thường gọi các hồ sơ vụ án mà Watson viết là giật gân và theo chủ nghĩa dân túy, cho thấy rằng chúng không báo cáo chính xác sự khách quan và tính "khoa học" trong công việc của anh:

Việc điều tra là, hay lẽ ra phải là, một môn khoa học chính xác, và phải được phản ánh với cùng một cung cách thản nhiên và vô cảm như thế. Anh lại muốn nó [Cuộc điều tra màu đỏ] nhuốm màu sắc chủ nghĩa lãng mạn, hiệu ứng tạo ra chẳng khác gì khi anh nhào nặn một chuyện tình hay chuyện bỏ nhà theo trai thành tiên đề thứ năm của Euclid. Một số tình tiết thực lẽ ra nên loại bỏ, hoặc chí ít khi đề cập tới cũng phải để ý sao cho nó có cảm giác tương xứng. Điểm duy nhất đáng nói trong vụ đó là quá trình suy luận mang tính phân tích cặn kẽ từ nguyên nhân đến kết quả mà nhờ đó tôi đã phá án thành công.[26]

Dù vậy, tình bạn với Watson là mối quan hệ quan trọng nhất của Holmes. Khi Watson bị trúng đạn, mặc dù vết thương có vẻ "khá hời hợt", vị bác sĩ cũng vẫn phải cảm động trước phản ứng của bạn mình:

Nó đáng giá một vết thương, thậm chí là nhiều vết thương, để biết được tận cùng lòng trung thành và tình yêu ẩn sau lớp mặt nạ lạnh lùng đó. Trong một giây lát, đôi mắt trong veo, rắn rỏi chợt mờ đi và đôi môi cương nghị thì đang run bần bật. Lần đầu tiên tôi được thấy một trái tim đi kèm một trí óc thật vĩ đại. Phát hiện này bù đắp đầy đủ tất cả những năm tháng cộng tác khiêm tốn và bất vụ lợi của tôi.[27]

Công việc
Khách hàng của Holmes rất đa dạng, từ các quốc vương hay chính phủ quyền lực nhất châu Âu, quý tộc, nhà công nghiệp giàu có cho đến những người làm nghề cầm đồ và các phó mẫu nghèo khó. Trong những câu chuyện đầu tiên, dù chỉ mới được vài người trong giới biết đến nhưng Holmes đã sớm cộng tác với Scotland Yard. Hồ sơ của Holmes ngày một dày hơn khi anh tiếp tục công việc và Watson thì cho xuất bản các câu chuyện của mình. Holmes nhanh chóng trở nên nổi tiếng trong vai trò một thám tử. Rất nhiều khách hàng tìm tới sự giúp đỡ của Holmes thay vì (hoặc ngoài) cảnh sát.[28] Watson viết, vào năm 1895, Holmes đã có "một lượng lớn việc làm".[29] Cảnh sát ngoài khu vực London cũng luôn yêu cầu Holmes giúp sức mỗi khi anh ở gần họ.[30] Thủ tướng[31] và vua Bohemia[32] đã đích thân đến thăm số 221B, phố Baker để nhờ Holmes hỗ trợ; tổng thống Pháp tặng anh Huân chương Bắc đẩu bội tinh vì bắt được một tên sát thủ;[33] vua Scandinavia là khách hàng của Holmes;[34] và Holmes cũng từng cứu viện Vatican ít nhất hai lần.[35] Vị thám tử từng làm thay chính phủ Anh trong các vấn đề an ninh quốc gia nhiều lần,[36] và từ chối phong tước hiệp sĩ "cho các dịch vụ mà có lẽ một ngày nào đó sẽ được mô tả".[37] Holmes không tích cực tìm kiếm danh vọng, thường xuyên bằng lòng để cho các cảnh sát giành hết công lao của mình.[38]

Điểm gián đoạn vĩ đại
Holmes and Moriarty wrestling at the end of a narrow path, with Holmes's hat falling into a waterfall
Holmes và Moriarty vật lộn tại đèo Reichenbach; vẽ bởi Sidney Paget
Tập truyện Sherlock Holmes đầu tiên được xuất bản từ năm 1887 đến năm 1893. Conan Doyle đã để Holmes chết trong trận chiến sau chót với bậc thầy tội phạm Giáo sư James Moriarty[39] trong "Vụ án cuối cùng" (xuất bản năm 1893, nhưng lấy bối cảnh năm 1891) vì ông cảm thấy rằng "tôi không nên dồn quá nhiều năng lượng văn học cho chỉ một kênh duy nhất."[40] Thế nhưng, phản ứng của công chúng đã khiến Doyle vô cùng ngạc nhiên. Những độc giả đau khổ liên tục gửi những lá thư sầu thảm cho The Strand Magazine, tạp chí này thì chịu một đòn khủng khiếp khi bị 20.000 người khiếu nại bằng việc hủy đăng ký.[41] Bản thân Conan Doyle cũng nhận được vô vàn thư phản đối, một phụ nữ thậm chí còn bắt đầu bức thư của mình bằng câu "Đồ súc vật".[41] Có giai thoại kể rằng khi nghe tin Holmes qua đời, người dân London đã tiếc thương tới mức đeo băng đen để tang cho anh. Không có nguồn tin đương thời nào kiểm chứng giai thoại trên, tài liệu tham khảo sớm nhất liên quan tới các sự kiện tương tự phải đến năm 1949 mới được biết đến.[42] Tuy nhiên, phải thừa nhận rằng những phản ứng được ghi lại của công chúng dành cho cái chết của Holmes không giống với bất kỳ phản ứng nào trước đây với các sự kiện hư cấu.[6]

Conan Doyle viết Con chó của dòng họ Baskerville (đăng nhiều kỳ vào năm 1901–02, với bối cảnh ngầm hiểu là trước khi Holmes qua đời) sau tám năm phải chống chọi với áp lực dư luận. Năm 1903, ông viết "Ngôi nhà trống không" lấy bối cảnh năm 1894. Holmes xuất hiện trở lại, giải thích cho Watson trong sự sửng sốt rằng mình đã giả chết để đánh lừa kẻ thù.[43] Sau "Ngôi nhà trống không", Conan Doyle vẫn thường xuyên viết tiếp các tác phẩm Sherlock Holmes cho đến năm 1927.
'''
# test_data_vf = {"data": test_data_vf}
save_data(test_data_vf, "test_mailong_data")


mailong_raw_datasets_vf = load_dataset("D:/WorkSpace/KhoaLuanTotNghiep/visquadv110.py")

mailong_raw_datasets_vf["train"] = mailong_raw_datasets_vf["train"].filter(lambda x: len(x["answers"]["text"]) == 1)
mailong_raw_datasets_vf["validation"] = mailong_raw_datasets_vf["validation"].filter(lambda x: len(x["answers"]["text"]) == 1)


mailong_validation_dataset_vf = mailong_raw_datasets_vf["train"].map(
    preprocess_validation_dataset,
    batched=True,
    remove_columns= mailong_raw_datasets_vf["train"].column_names,
)
print(mailong_validation_dataset_vf)


metric = evaluate.load(args_metric)

mailong_validation_set_vf = mailong_validation_dataset_vf.remove_columns(["example_id", "offset_mapping"])
mailong_validation_set_vf.set_format("torch")

mailong_eval_dataloader_vf = DataLoader(
    mailong_validation_set_vf,
    collate_fn=default_data_collator,
    batch_size=args_batch_size
)

for item in mailong_eval_dataloader_vf:
    print(item)

trained_model = AutoModelForQuestionAnswering.from_pretrained(args_output_dir)
trained_model.to(device)

trained_model.eval()
start_logits = []
end_logits = []
print("Evaluation!")
for batch in tqdm(mailong_eval_dataloader_vf):
#     print(batch)
#     # Chuyển đổi batch sang cùng một thiết bị mà mô hình đã được di chuyển đến
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
print(f"Epoch :", metrics)

import os
from time import sleep

def copy_and_delete_file(source_file, destination_file):
    # Đọc nội dung của file nguồn
    with open(source_file, 'r') as file:
        file_content = file.read()
    os.remove(source_file)
    sleep(1)
    # Lưu nội dung vào file đích
    
    with open(destination_file, 'w') as file:
        file.write(file_content)

    # Xóa file nguồn

# Đường dẫn của file nguồn và file đích
source_file_path = 'D:/WorkSpace/KhoaLuanTotNghiep/visquadv110.py'
destination_file_path = 'D:/WorkSpace/KhoaLuanTotNghiep/visquadv110.py'

# Sao chép nội dung từ file nguồn sang file đích và xóa file nguồn
copy_and_delete_file(source_file_path, destination_file_path)
