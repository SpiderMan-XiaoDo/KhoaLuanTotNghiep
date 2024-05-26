import shutil
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
source_file_path = 'D:/WorkSpace/KhoaLuanTotNghiep/test_3.py'
destination_file_path = 'D:/WorkSpace/KhoaLuanTotNghiep/test_3.py'

# Sao chép nội dung từ file nguồn sang file đích và xóa file nguồn
copy_and_delete_file(source_file_path, destination_file_path)
