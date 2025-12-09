# Example 3.17 Least Squares Fitting
import numpy as np
import scipy.optimize as optimization

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([100, 90, 60, 30, 10, 1])


# กำหนดฟังก์ชันที่จะใช้ปรับให้เข้ากับข้อมูล
def func(x, a, b, c):
    return a * np.exp(-b * x) + c


# ทำการ Least Squares Fitting
# popt: ค่าพารามิเตอร์ที่ดีที่สุด (a, b, c)
# pcov: เมทริกซ์ความแปรปรวนร่วม (Covariance Matrix)
result = optimization.curve_fit(func, x, y, p0=[100, 0.5, 0], maxfev=5000)
popt = result[0]
pcov = result[1]

print("Best fit a b c: ", popt)
print("Best fit covariance: ", pcov)
