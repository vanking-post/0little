import step10SVM
import step10LSTM
import gc
from step07pca import save_dir
#save_dir = r"E:\0little\read\CQSkyEyedata5"
#SVM模型训练
# step10SVM.train_svm(save_dir)
#LSTM模型训练
step10LSTM.train_lstm(save_dir)