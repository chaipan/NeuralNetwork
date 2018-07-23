import numpy as np

def rand_arr(a, b, *args):
    # 设置相同的seed时，每次生成的随机数相等
    np.random.seed(0)
    randvar = np.random.rand(*args)
    print(randvar)
    return np.random.rand(*args) * (b - a) + a

if __name__ == "__main__":
    var = rand_arr(-0.1, 0.1, 2, 4)
    print(var)