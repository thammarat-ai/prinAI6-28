#Example 3.1 Python SVM Classifications
from sklearn import svm

# X: ข้อมูลคุณลักษณะ (Features)
# [[ความสูง (ซม.), น้ำหนัก (กก.), ขนาดรองเท้า (UK)]]
X = [[170, 70, 10], [180, 80, 12], [170, 65, 8], [160, 55, 7]] 

# y: ป้ายกำกับ (Labels) หรือ คลาส (Classes) ที่ทราบแล้ว
# 0: ชาย (Male), 1: หญิง (Female)
y = [0, 0, 1, 1] 

# 1. สร้างตัวจำแนก (Classifier)
clf = svm.SVC()

# 2. ฝึกฝนโมเดล (Training)
clf.fit(X, y)

# 3. ทำการทำนาย (Prediction) สำหรับข้อมูลตัวอย่างใหม่
p = clf.predict([[160, 60, 7]])

# 4. แสดงผลลัพธ์
print(p)