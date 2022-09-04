import os
import pandas as pd

os.makedirs(os.path.join('../d2l', 'data'), exist_ok=True)
data_file = os.path.join('../d2l', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每⾏表⽰⼀个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
print(outputs)

# 利用pandas实现one hot encode的方式
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
