#Example 3.7 Naive Bayes Iris
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# โหลดข้อมูล Iris โดยตรง (X=Features, y=Labels)
X, y = load_iris(return_X_y=True) 
print(X) # แสดงข้อมูลคุณลักษณะ (4 features: Sepal length/width, Petal length/width)

# สร้างและฝึกฝนตัวจำแนก
clf = GaussianNB() 
clf.fit(X, y)

# ทำนายผลสำหรับตัวอย่างใหม่ (Sepal length=5.0, width=3.4, Petal length=1.5, width=0.4)
p = clf.predict([[5.0, 3.4, 1.5, 0.4]]) 
print(p)