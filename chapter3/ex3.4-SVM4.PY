#Example 3.4 Python SVM Iris URL Classifications 
from sklearn import svm, datasets
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

# 0 . กำหนดตำแหน่งไฟล์ CSV ที่มีข้อมูล Iris จาก GitHub
csv_url = 'https://raw.githubusercontent.com/thammarat-ai/prinAI6-28/refs/heads/main/chapter3/iris.csv'

# 1. อ่านข้อมูลจากไฟล์ CSV ด้วย Pandas
# กำหนดชื่อคอลัมน์เอง และข้ามแถวแรก (skiprows=1)
df = pd.read_csv(csv_url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'], skiprows=1)

# การวิเคราะห์ข้อมูลเบื้องต้น (EDA)
print(df.shape) 
print(df.head(10)) 
print(df.tail(10)) 
print(df.describe())

# การจัดการข้อมูลสูญหาย (Missing Data)
print(df.isna().sum().sum()) 
df = df.dropna()

# การสำรวจความหลากหลายของป้ายกำกับ (Label Distribution)
print(df.groupby('species').size())

# การสร้างภาพข้อมูล (Data Visualization)
df.hist() 
pyplot.show() 
scatter_matrix(df) 
pyplot.show() 

# การเตรียมข้อมูลสำหรับ SVM และการฝึกฝน/ทำนาย
X = df.values[:,:2] 
s = df['species']
d = dict([(y,x) for x,y in enumerate(sorted(set(s)))]) 
y = [d[x] for x in s]
clf = svm.SVC() 
clf.fit(X, y)
p = clf.predict([[5.4, 3.2]])
print(p)