#Example 3.5 Breast Cancer Data
from sklearn.datasets import load_breast_cancer

# โหลดชุดข้อมูลมะเร็งเต้านม
cancer = load_breast_cancer() 

# แสดงชื่อคุณลักษณะ (Feature Names)
print(cancer['feature_names']) 

# แสดงข้อมูลคุณลักษณะ (Feature Data)
print(cancer['data']) 

# แสดงชื่อป้ายกำกับเป้าหมาย (Target Names)
print(cancer.target_names)