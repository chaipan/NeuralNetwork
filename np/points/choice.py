"""
np.random.choice函数，从给出数据中以指定概率生成指定大小的数据
"""

import numpy as np
"""
    def choice(a, size=None, replace=True, p=None):
    a:选择值（候选值）
    size:输出大小
"""
# 不带p
a = np.random.choice([1,2,3], size=10)
print(a)
# [1 3 1 1 3 2 2 2 1 3]
# 带p,p的和为1
a = np.random.choice([1,2,3], size=10, p=[0,0,1])
print(a)
# [3 3 3 3 3 3 3 3 3 3]
