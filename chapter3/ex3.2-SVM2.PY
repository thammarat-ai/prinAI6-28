#Example 3.2 Python SVM Iris Classifications
from sklearn import svm, datasets

# โหลดชุดข้อมูล Iris
iris = datasets.load_iris()

# X: เลือกคุณลักษณะ 2 ตัวแรก (Sepal length and Sepal width)
X = iris.data[:, :2] 

# y: ป้ายกำกับเป้าหมาย (0: Setosa, 1: Versicolour, 2:Virginica)
y = iris.target 

print(y)

# 1. สร้างและฝึกฝนตัวจำแนก (Classifier Training)
clf = svm.SVC()
clf.fit(X, y)

# 2. ทำนายดอกไม้จากความยาวและความกว้างของกลีบเลี้ยงที่กำหนดให้
p = clf.predict([[5.4, 3.2]])

print(p)