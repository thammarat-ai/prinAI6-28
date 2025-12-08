#Example 3.3 Python SVM Iris CSV Classifications
from sklearn import svm, datasets
import pandas as pd

# 1. อ่านข้อมูลจากไฟล์ CSV ด้วย Pandas
df = pd.read_csv('iris.csv') 

# 2. X: เลือกคุณลักษณะ 2 ตัวแรก (Sepal length, Sepal width)
# .values เพื่อแปลง DataFrame เป็น NumPy array
X = df.values[:,:2] 

# 3. เตรียมป้ายกำกับ (Labels)
s = df['species'] # ดึงคอลัมน์ชื่อสปีชีส์ (ข้อความ)
d = dict([(y,x) for x,y in enumerate(sorted(set(s)))]) # สร้าง Dictionary เพื่อแปลงชื่อสปีชีส์เป็นตัวเลข (0, 1, 2)
y = [d[x] for x in s] # แปลงคอลัมน์ species เป็นป้ายกำกับตัวเลข (y)

# 4. สร้างและฝึกฝนตัวจำแนก (Classifier Training)
clf = svm.SVC() 
clf.fit(X, y)

# 5. ทำนาย
p = clf.predict([[5.4, 3.2]])
print(p)