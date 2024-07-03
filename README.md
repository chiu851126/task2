# task2
Task 1: General Questions

1.	What is your preferred language when building predictive models and why?
Python,因為使用起來淺顯易懂，在撰寫操作運行上方便許多，及時出現程式錯誤，能更有效的修正，且在應用領域上也很廣泛，像是網路爬蟲、網頁開發、人工智慧、資料科學/分析及視覺化的呈現等，都是能在Python程式語言中，讓開發人員能夠更有效的完成更多的工作。
2.	Provide an example of when you used SQL to extract data.
SQL,顧名思義是一種結構化的程式語言，用於在關聯式資料庫中儲存和處理資訊的程式設計語言。例如:1.產生資料庫中的資料表，2.定義資料表欄位和資料型態，3.建立表格之間有無關聯性，4. 將資料進行處理有四種基本查詢動作，即新增、修改、刪除、查詢，5.將資料進行統計。基本語法指令：SELECT、FROM、WHERE，以下是提取資料的範例：

Employee資料表：
Fname	Lname	Ssn	Bdate	Address	Sex	Salary	Super_ssn	Dno
John	Smith	123456789	1965-01-09	731 Fondren, Houston, TX	M	30000	333445555	5
Franklin	Wong	333445555	1955-12-08	638 Voss, Houston, TX	M	40000	888665555	5
Alicia	Zelaya	999887777	1968-01-19	3321 Castle, Spring, TX	F	25000	987654321	4
Jennifer	Wallace	987654321	1941-06-20	291 Berry, Bellaire, TX	F	43000	888665555	4

例如1：要找出所有顧客的所有欄位和所有屬性。

Select *
From Employee;

Fname	Lname	Ssn	Bdate	Address	Sex	Salary	Super_ssn	Dno
John	Smith	123456789	1965-01-09	731 Fondren, Houston, TX	M	30000	333445555	5
Franklin	Wong	333445555	1955-12-08	638 Voss, Houston, TX	M	40000	888665555	5
Alicia	Zelaya	999887777	1968-01-19	3321 Castle, Spring, TX	F	25000	987654321	4
Jennifer	Wallace	987654321	1941-06-20	291 Berry, Bellaire, TX	F	43000	888665555	4

例如2：employee 表格中的所有女性，選取並顯示其姓名、社會安全碼、薪資

SELECT fname, lname, ssn, salary
FROM employee
WHERE sex = 'F'

Fname	Lname	Ssn	Salary
Alicia	Zelaya	999887777	25000
Jennifer	Wallace	987654321	43000

3.	Give an example of a situation where you disagreed upon an idea or solution design with a co-worker.  How did you handle the case?
當我與同事之間對於同個專案案件及設計有分岐時，1.我會提議先彼此冷靜，然後自我反思，幾分鐘後，針對案子問題進行討論，2.如果還是有分歧時，請主管來提出意見，得到主管的認可或者是通過表決方式進行處理，彼此都能達成共識以及適當妥協。

4.	What are your greatest strengths and weaknesses and how will these affect your performance here?
最大優點：1. 對於自己所愛的專業領域，勇於挑戰新鮮事物，帶給公司更多的效益。2.在不同環境下，吸收更多知識，獲得收穫及成長。3. 我做事仔細並堅持到底。在我處理專案時，我會持續追蹤細節，並嚴格要求這些事項在期限前完成。4. 我是很有同理心的人，並懂得去理解別人所想的，使得他們覺得自己的意見受到肯定及重視。

最大缺點：1.時常覺得自己努力不夠，感到懊惱。2.我是一位心思縝密的人，怕因為某個專案太趕，無法達預期所望。3.我個人是一位內向的人，時常怕麻煩到人，但我還是會鼓勵自已勇於面對及溝通，不管是工作以及生活中，讓自己開放一點，終究會得到不同的收穫。

Task 2: Python model development

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



