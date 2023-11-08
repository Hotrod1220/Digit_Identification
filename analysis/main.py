from convolution_28x28 import Convolution28
from convolution_20x20 import Convolution20
from divide import DivideConquer
from divide_inner import DivideConquerInner

if __name__ == '__main__':
    convolution_28 = Convolution28()
    convolution_20 = Convolution20()
    divide_conquer = DivideConquer()
    divide_conquer_real = DivideConquerInner(scanned = True, folder='scan')
    divide_conquer_inner = DivideConquerInner(folder='nxn_inner')

    # Task B with digits of size 28x28
    # convolution_28.analyze()
    
    # Task B with digits of size 20x20 
    # convolution_20.analyze()

    # Task C
    # divide_conquer.analyze()

    # Task C with Scanned Images
    # divide_conquer_real.analyze()

    # Task C with digits able to be in other digits
    divide_conquer_inner.analyze()
