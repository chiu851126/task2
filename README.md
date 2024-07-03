# task2
訓練資料
資料前處理
類別型態:
Geography:類別型態轉數值(nominal)，使用onehotencoding進行轉換
Gender:類別型態轉數值型態(binary)，使用onehotencoding進行轉換

數值型態:
CreditScore:判斷是否有離群值(Outlier)，運算結果有240個離群值，將欄位刪除，處理完成後，由於數值過大，使用資料標準化進行處理
Age:判斷是否有離群值(Outlier)，運算結果有6394個離群值，將欄位刪除，處理完成後，由於數值過大，使用資料標準化進行處理
Tenure:因數值過大，使用資料標準化進行處理
Balance:因數值過大，使用資料標準化進行處理
EstimatedSalary:因數值過大，使用資料標準化進行處理
Exited:由於資料筆數相差太遠，使得資料不平衡，所以使用SMOTE生成進行訓練

資料筆數:原本165034筆，刪完離群值(Age:6394,CreditScore:240)，總共158400筆
首先將訓練資料欄位裡的'id', 'CustomerId', 'Surname', 'Exited'進行移除
資料切分:80%訓練，20%測試(8:2)
使用五種機器學習演算法:LogisticRegression、KNN、RandomForest、XGB、LightGBM
交叉驗證(CV):10倍
評估指標:精確率、召回率、F1測量

SMOTE生成出來結果:
LogisticRegression:
驗證準確率(CV=10):0.7491
precision: 0.4352
recall: 0.7386
F1: 0.5477

KNN:
驗證準確率(CV=10):0.8596
precision: 0.4560
recall: 0.7265
F1: 0.5603

RandomForest:
驗證準確率(CV=10):0.9048
precision: 0.6205
recall: 0.6124
F1: 0.6164

XGB:
驗證準確率(CV=10):0.8847
precision: 0.6573
recall: 0.6183
F1: 0.6372

LightGBM:
驗證準確率(CV=10):0.8966
precision: 0.6822
recall: 0.5984
F1: 0.6376

結論:加入SMOTE生成資料，使得覆蓋率提高，透過PR曲線圖分析，顯然得出LightGBM模型>XGB模型>RandomForest模型>KNN模型>LogisticRegression模型。![alt text](<Precision-Recall Curve_smote.png>)
-------------------------------------------------------------------------------------------------
測試資料
類別型態:
Geography:類別型態轉數值(nominal)，使用onehotencoding進行轉換
Gender:類別型態轉數值型態(binary)，使用onehotencoding進行轉換

數值型態:
CreditScore:up 因數值過大，使用資料標準化進行處理
Age:因數值過大，使用資料標準化進行處理
Tenure:因數值過大，使用資料標準化進行處理
Balance:因數值過大，使用資料標準化進行處理
EstimatedSalary:因數值過大，使用資料標準化進行處理

資料筆數:總共110023筆
首先將訓練資料欄位裡的'id', 'CustomerId', 'Surname'進行移除
資料切分:80%訓練，20%測試(8:2)
使用五種機器學習演算法預測:LogisticRegression、KNN、RandomForest、XGB、LightGBM

預測之結果:預測'Exited'的機率
LogisticRegression模型:[text](y_pred_Exited_LR.csv)之結果
KNN模型:[text](y_pred_Exited_KNN.csv)之結果
RandomForest模型:[text](y_pred_Exited_RF.csv)之結果
XGB模型:[text](y_pred_Exited_XGB.csv)之結果
LightGBM模型:[text](y_pred_Exited_LightGBM.csv)之結果



