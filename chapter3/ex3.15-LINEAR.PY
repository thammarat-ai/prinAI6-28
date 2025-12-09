# Example 3.15 Linear Regression
import matplotlib.pyplot as plt
from scipy import stats

x = [0, 1, 2, 3, 4]
y = [3, 5, 5, 6, 7]

# คำนวณการถดถอยเชิงเส้น
slope, intercept, r, p, std_err = stats.linregress(x, y)

print("slope: ", slope)
print("intercept: ", intercept)


# ฟังก์ชันเส้นตรง f(x) = ax + b
def myfunc(x):
    return slope * x + intercept


# คำนวณจุดบนเส้นตรงโดยใช้ค่า x เดิม
mymodel = list(map(myfunc, x))

# สร้างกราฟ
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
