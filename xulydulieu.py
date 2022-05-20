from sklearn.model_selection import train_test_split
from pyvi import ViTokenizer  
import pickle 
import os 
import re

#Load stopwords
with open('./word/stopwords.txt', 'r', encoding="Utf-8") as f:
    stopwords = f.read()

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
                lines = re.sub(r'\\xa0',' ',str(lines))
                #Xóa ký tự đặc biệt
                document = re.sub(r'\W', ' ', str(lines))
                #Xóa các số
                document =  ''.join([i for i in document if not i.isdigit()])
                #Xóa khoản trắng thừa
                document = re.sub(r'\s+', ' ', document)
                #Xóa các ký tự đơn
                document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
                #In thường
                document = document.lower() 
                #Tách từ
                lines = ViTokenizer.tokenize(document)
                #Xóa stopwords
                lines = ' '.join([w for w in lines.split() if not w in stopwords])

                X.append(lines)
                y.append(path)

    return X, y

#Load datasets
data_path = os.path.join('./vnx-dev')
X_data, y_data = get_data(data_path)

#Chia tập test, train, val
X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=0.2, train_size=0.8,random_state=100, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.25,train_size =0.75,random_state=100, shuffle=True)

#Lưu tập test
pickle.dump(X_test,  open('./word/X_test.pkl', 'wb'))
pickle.dump(y_test,  open('./word/y_test.pkl', 'wb'))
#Lưu tập train
pickle.dump(X_train, open('./word/X_train.pkl', 'wb'))
pickle.dump(y_train, open('./word/y_train.pkl', 'wb'))
#Lưu tập val
pickle.dump(X_val, open('./word/X_val.pkl', 'wb'))
pickle.dump(y_val, open('./word/y_val.pkl', 'wb'))

print(X_data[:1])
