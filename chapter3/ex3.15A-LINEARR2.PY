# Example 3.15a Linear Regression
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

x = [0, 1, 2, 3, 4]
y = [3, 5, 5, 6, 7]

x1 = sm.add_constant(x)  # เพิ่มค่าคงที่ (Intercept) เข้าไปในข้อมูล X
model = sm.OLS(y, x1)  # สร้างโมเดล OLS (Ordinary Least Squares)
results = model.fit()  # ฝึกฝนโมเดล
print(results.params)  # แสดงค่าพารามิเตอร์ (Slope และ Intercept)
print(results.summary())  # แสดงตารางสรุปผลลัพธ์ทางสถิติอย่างละเอียด

y_pred = results.predict(x1)  # คาดการณ์ค่า Y จากโมเดล
plt.scatter(x, y)  # พล็อตจุดข้อมูลจริง
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, y_pred, "r")  # พล็อตเส้นถดถอย (สีแดง)
plt.show()
