# Example 3.18 Multiple Linear Regression
from sklearn import linear_model
import numpy as np

# อาร์เรย์ X (ตัวแปรอิสระ 3 ตัว: X1, X2, X3)
x = np.array([[0, 3, 5], [1, 4, 6], [2, 5, 7], [3, 6, 8], [4, 7, 9]])

# อาร์เรย์ Y (ตัวแปรตาม 1 ตัว)
y = np.array([3, 5, 5, 6, 7])

reg = linear_model.LinearRegression()
reg.fit(x, y)  # ฝึกฝนโมเดล

print("Coefficients: \n", reg.coef_)  # แสดงค่าสัมประสิทธิ์ (Beta)
print("Intercept: \n", reg.intercept_)  # แสดงค่าจุดตัดแกน (Beta 0)

# ทำการคาดการณ์สำหรับข้อมูลใหม่ (X1=5, X2=8, X3=10)
pred = reg.predict([[5, 8, 10]])
print("Prediction: \n", pred)