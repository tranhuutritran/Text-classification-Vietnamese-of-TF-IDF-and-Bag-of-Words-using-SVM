from sklearn.metrics import classification_report, confusion_matrix
import pickle

#Load model đã huấn luyện
model_svm_tfidf = pickle.load(open('./word/model_svm_tfidf.pkl'ư, 'rb'))
model_svm_bag   = pickle.load(open('./word/model_svm_bag.pkl', 'rb'))

#Load dữ liệu tập test
X_test_tfidf = pickle.load(open('./word/X_test_tfidf.pkl', 'rb'))
X_test_bag   = pickle.load(open('./word/X_test_bag.pkl', 'rb'))
y_test       = pickle.load(open('./word/y_test.pkl', 'rb'))

#Đưa dữ liệu test vào model phân lớp
y_pred_tfidf = model_svm_tfidf.predict(X_test_tfidf)
y_pred_bag   = model_svm_bag.predict(X_test_bag)

#Đánh giá IF-TDF_SVM
print(classification_report(y_test, y_pred_tfidf))
#Số nhãn đoán đúng
print(confusion_matrix(y_test, y_pred_tfidf))

#Đánh giá BAG_SVM
print(classification_report(y_test, y_pred_bag))
print(confusion_matrix(y_test, y_pred_bag))
