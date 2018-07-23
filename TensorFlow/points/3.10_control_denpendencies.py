import TensorFlow as tf


"""
    想要指定某些操作执行的依赖关系
    TensorFlow.control_dependencies(control_inputs)返回控制依赖的上下文管理器，
    使用with关键字可以让在这个上下文环境中的操作都在control_input中执行
    
    
    TF还可以协调多个数据流，在存在依赖节点的场景下非常有用，例如节点B要读取模型参数θ更新后的值，
    而节点A负责更新参数θ，则节点B必须等节点A完成后才能执行，否则读取的参数θ为更新前的数值，
    这时需要一个运算控制器。接口函数如下，TensorFlow.control_dependencies函数可以控制多个数据流执行完成
    后才能执行接下来的操作，通常与tf.group函数结合使用。
"""


