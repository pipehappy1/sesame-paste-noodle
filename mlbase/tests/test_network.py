import theano
import theano.tensor as T
import mlbase.network as N
import numpy as np
from mlbase.layers import *
import pytest

def test_nextLayerSeq():
    n = N.Network()

    n.setInput(RawInput((1, 28, 28)))
    n.append(Flatten())
    n.append(FullConn(feature_map_multiplier=2))
    n.append(Elu())
    n.append(FullConn(output_feature=10))
    n.append(output.SoftMax())

    g = n.nextLayer()
    assert issubclass(type(next(g)), RawInput)
    assert issubclass(type(next(g)), Flatten)
    assert issubclass(type(next(g)), FullConn)
    assert issubclass(type(next(g)), Elu)
    assert issubclass(type(next(g)), FullConn)
    assert issubclass(type(next(g)), output.SoftMax)
    with pytest.raises(Exception) as e:
        next(g)

def test_nextLayerDiamond():

    n = N.Network()
    
    inputLayer = RawInput((1, 28, 28))
    n.setInput(inputLayer)
    flatten = inputLayer.followedBy(Flatten())
    full1 = flatten.followedBy(FullConn(feature_map_multiplier=2))
    full2 = flatten.followedBy(FullConn(feature_map_multiplier=2))
    concat = Concat().follow(full1, full2)
    full3 = concat.followedBy(FullConn(feature_map_multiplier=2))
    
    g = n.nextLayer()
    assert next(g) == inputLayer
    assert next(g) == flatten
    assert next(g) == full2
    assert next(g) == full1
    assert next(g) == concat
    assert next(g) == full3
    with pytest.raises(Exception) as e:
        next(g)

    
def test_predictBatchSize():
    """
    Test batch size works for perdictor.
    """
    n = N.Network()
    n.batchSize = 2

    n.inputSizeChecker = [1,1]
    
    x = T.fmatrix()
    y = T.switch(T.gt(x,0), 1, 0)
    f = theano.function([x], y, allow_input_downcast=True)
    n.predicter = f

    tx = np.array([[-0.27540332], [-0.76737626], [ 0.84122449], [-1.96092991], [-0.44198351],
                   [ 0.79166672], [ 0.87340424], [ 0.04555511], [-2.11510706], [-0.10966502],
                   [ 0.54762297], [-1.56990211], [-0.61545427], [ 1.11211698], [-0.66220848],
                   [ 0.11964702], [-2.15263133], [-1.8672312 ], [ 0.22093941], [-0.46957548]])
    ty = np.array([[0], [0], [1], [0], [0],
                   [1], [1], [1], [0], [0],
                   [1], [0], [0], [1], [0],
                   [1], [0], [0], [1], [0]])
    tlen = 20

    assert (ty == n.predict(tx)).all()
    assert (ty[:(tlen-1),:] == n.predict(tx[:(tlen-1),:])).all()


#def test_predictWithIntermediaResult():
#    """
#    Test to see we can see intermediate result after each layer.
#    """
#
#    class Linear2d(Layer):
#        def __init__(self):
#            super(Linear2d, self).__init__()
#
#        def forwardSize(self, inputsize):
#            isize = inputsize[0]
#            return [(isize[0], 2,)]
#
#    class Linear2da(Linear2d):
#        def __init__(self):
#            super(Linear2da, self).__init__()
#            self.w = theano.shared(np.array([[1, 2],[3, 4]]), borrow=True)
#                
#        def predictForward(self, inputtensor):
#            inputimage = inputtensor[0]
#            return (T.dot(inputimage, self.w),)
#
#    class Linear2db(Linear2d):
#        def __init__(self):
#            super(Linear2db, self).__init__()
#            self.w = theano.shared(np.array([[-2.0, 1.0],[1.5, -0.5]]), borrow=True)
#            
#        def predictForward(self, inputtensor):
#            inputimage = inputtensor[0]
#            return (T.dot(inputimage, self.w),)
#        
#    n = N.Network()
#    n.setInput(RawInput((2,)))
#    n.append(Linear2da(), "output1")
#    n.append(Linear2db(), "output2")
#
#    n.build()
#
#    tx = np.array([[ 1.38921142,  0.57967604],
#                   [-0.56795221,  1.38135903],
#                   [-0.30971383, -1.06001774],
#                   [-1.70132043,  1.78895373],
#                   [-0.59605122,  0.8748537 ],
#                   [-0.05554206, -0.62843449]])
#    ty = tx
#
#    assert (np.abs(ty - n.predict(tx)) < 0.001).all()


