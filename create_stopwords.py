from pyvi import ViTokenizer  
import os 
import re

words_counts = {}

def get_data(folder_path):
    # X là nội dung văn bản
    # y là nhãn/thể loại của văn bản đó
    X = []       
    y = []  
    #Trả về danh sách tên các chủ đề trong folder_path
    dirs = os.listdir(folder_path)  
    for path in dirs:
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in file_paths:
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-8", errors='ignore') as f:
                lines = f.readlines()
                #Xóa \xa0 khỏi chuỗi
                document = re.sub(r'\\xa0',' ',str(lines))
                #Xóa ký tự đặc biệt
                document = re.sub(r'\W', ' ', str(document))
                #Xóa các số
                document =  ''.join([i for i in document if not i.isdigit()])
                #Xóa khoản trắng thừa
                document = re.sub(r'\s+', ' ', document)
                #Xóa các ký tự đơn
                document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
                #In thường
                lines = document.lower() 
                #Tách từ
                lines = ViTokenizer.tokenize(document)
                #Tạo words_counts
                for word in lines.split():
                    if word not in words_counts:
                        words_counts[word] = 1
                    else:
                        words_counts[word] += 1 
                X.append(lines)
                y.append(path)

    return X, y

#Load datasets
data_path = os.path.join('./vnx-dev')
X_data, y_data = get_data(data_path)

#Lấy 100 từ xuất hiện nhiều nhất làm stopwords
sorted_count = sorted(words_counts, key=words_counts.get, reverse=True)[:100]
with open('./word/stopwords.txt', 'w', encoding='utf8') as fp:
        for word in sorted_count:
            fp.write(word + '\n')
            print(word, words_counts[word])

