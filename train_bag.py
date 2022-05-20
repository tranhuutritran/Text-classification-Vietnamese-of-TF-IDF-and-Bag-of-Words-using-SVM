from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
import pickle
import time

#Load tập train, val, test
X_train = pickle.load(open('./word/X_train.pkl', 'rb'))
y_train = pickle.load(open('./word/y_train.pkl', 'rb'))
X_val   = pickle.load(open('./word/X_val.pkl', 'rb'))
y_val   = pickle.load(open('./word/y_val.pkl', 'rb'))
X_test  = pickle.load(open('./word/X_test.pkl', 'rb'))

start_time = time.time()

#Bag_features
def bag_features(X_train, X_val, X_test):
    bag_vectorizer = CountVectorizer(analyzer='word',binary=True)
    X_train_bag = bag_vectorizer.fit_transform(X_train)
    X_val_bag   = bag_vectorizer.transform(X_val)
    X_test_bag  = bag_vectorizer.transform(X_test)
    return X_train_bag, X_val_bag, X_test_bag

X_train_bag, X_val_bag, X_test_bag = bag_features(X_train, X_val, X_test)
#print(X_train_bag.toarray())

#Huấn luyện mô hình phân lớp SVM
model_svm_bag = svm.SVC(C=2.0, kernel='linear', gamma='auto').fit(X_train_bag, y_train)

#Đánh giá accuracy mô hình phân lớp SVM
y_val_predict_bag = model_svm_bag.predict(X_val_bag)
print("Bag_SVM, Accuracy:",accuracy_score(y_val, y_val_predict_bag)*100)

train_time = time.time() - start_time
print('Thời gian đào tạo', train_time, 'seconds.')

#Lưu model huấn luyện SVM
pickle.dump(model_svm_bag, open('./word/model_svm_bag.pkl', 'wb'))
#Lưu tập X_test_bag để thực nghiệm
pickle.dump(X_test_bag, open('./word/X_test_bag.pkl', 'wb'))

