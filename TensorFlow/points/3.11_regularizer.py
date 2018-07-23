import TensorFlow as tf
from TensorFlow.contrib import layers
"""
    我们管R_emp(f)叫做经验风险，管上面我们思维导图的过程叫做正则化，所以顺其自然的管r(d)叫做正则化项，
    然后管R_emp(f)+r(d) 叫做结构风险，所以顺其自然的正则化就是我们将结构风险最小化的过程，它们是等价的。
    1范数，L_1正则化,向量元素的绝对值之和,||w||1
    2范数，L_2正则化，绝对值的平方和在开方（1/2）||w||2^2
    
"""

def L1_regularizer(arg = None):
    # l1正则化，lamda||w||1,前面的lamda为超参数
    l1_regu = layers.l1_regularizer(scale=0.01)
