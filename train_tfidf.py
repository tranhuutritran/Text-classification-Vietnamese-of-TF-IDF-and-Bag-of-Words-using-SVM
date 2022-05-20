from sklearn.feature_extraction.text import TfidfVectorizer
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

#Tfidf_features
def tfidf_features(X_train, X_val, X_test):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word',binary=True)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf   = tfidf_vectorizer.transform(X_val)
    X_test_tfidf  = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_val_tfidf, X_test_tfidf

X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf_features(X_train, X_val, X_test)
#print(X_train_tfidf.toarray())

#Huấn luyện mô hình phân lớp SVM
model_svm_tfidf = svm.SVC(C=2.0, kernel='linear', gamma='auto').fit(X_train_tfidf, y_train)

#Đánh giá mô hình phân lớp SVM
y_val_predict_tfidf = model_svm_tfidf.predict(X_val_tfidf)
print("Tfidf_SVM, Accuracy:",accuracy_score(y_val, y_val_predict_tfidf)*100)

train_time = time.time() - start_time
print('Thời gian đào tạo', train_time, 'seconds.')

#Lưu model huấn luyện SVM
pickle.dump(model_svm_tfidf, open('./word/model_svm_tfidf.pkl', 'wb'))
#Lưu tập X_test_tfidf để thực nghiệm
pickle.dump(X_test_tfidf, open('./word/X_test_tfidf.pkl', 'wb'))
