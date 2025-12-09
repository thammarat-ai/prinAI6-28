# Example 3.19 Logistic Regression
import numpy as np
from sklearn.linear_model import LogisticRegression

# ข้อมูลตัวแปรอิสระ (Features)
X = np.array([[0], [1], [2], [3], [4], [5]])
# ข้อมูลป้ายกำกับประเภท (Labels)
y = np.array([1, 2, 3, 30, 32, 31])

# สร้างและฝึกฝนโมเดล Logistic Regression
clf = LogisticRegression(random_state=0).fit(X, y)

# 1. ทำนายประเภทของตัวอย่างใหม่ (X=6)
print(clf.predict([[6]]))

# 2. ทำนายความน่าจะเป็นของตัวอย่างใหม่ (X=6) สำหรับแต่ละประเภท
print(clf.predict_proba([[6]]))

# 3. คำนวณความแม่นยำของโมเดลบนชุดข้อมูลฝึกสอน
print(clf.score(X, y))
