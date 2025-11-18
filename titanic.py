import pandas as pd
import numpy as np

#from sklearn.model_selection import train_test_split用于划分数据集 但是train和test已经给出就无需划分了
from sklearn.model_selection import cross_val_score#交叉验证
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score用于预测准确率（预测正确样本占总样本数的比例）

#read data      
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('submission.csv')

#preprocess data
#turn sex into labelcode
train['Sex']=LabelEncoder().fit_transform(train['Sex'])
test['Sex']=LabelEncoder().fit_transform(test['Sex'])
for df in[train,test]:
    df['Embarked']=df['Embarked'].fillna('S')
    df['Embarked']=LabelEncoder().fit_transform(df['Embarked'])
    df['Fare']=df['Fare'].fillna(df['Fare'].median())
    df['Age']=df['Age'].fillna(df['Age'].median())

train['Family']=train['SibSp']+train['Parch']
test['Family']=test['SibSp']+test['Parch']

features=['Pclass','Sex','Age','Fare','Embarked','Family']
x=train[features]
y=train['Survived']
x_test=test[features]

#train model
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)

scores=cross_val_score(model,x,y,cv=5,scoring='accuracy')#cv是折数 
print(f"Cross-validated Accuracy:{scores.mean():.4f}+{scores.std():.4f}")

#predict
preds=model.predict(x_test)
submission['Survived']=preds
submission.to_csv('final1.csv',index=False)
submission.head()