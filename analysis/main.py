from convolution_28x28 import Convolution28
from convolution_20x20 import Convolution20
from divide import DivideConquer

if __name__ == '__main__':
    convolution_28 = Convolution28()
    convolution_20 = Convolution20()
    divide_conquer = DivideConquer()

    # Task B with digits of size 28x28
    # convolution_28.analyze()
    
    # Task B with digits of size 20x20 
    # convolution_20.analyze()

    # Task C
    divide_conquer.analyze()
