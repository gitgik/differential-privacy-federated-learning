

```python
import torch
```


```python
def activation(x):
    """Create a sigmoid activation function.
    Good for outputs that fall between 0 and 1. (probability)
    args x: a torch tensor.
    """
    return 1/(1 + torch.exp(-x))
    
```


```python
# generate some data
# set some random seed so that the result is predictatble
data = torch.manual_seed(7) 

# set some features to 5 random variables
# 2-dimensional matrix/tensor of 1 row and 5 columns
features = torch.randn((1,5))

# set weights
weights = torch.randn_like(features)

# set true bias term
bias = torch.randn((1,1))
```


```python
# calculate y output
# y = (weights.features + bias)

x = torch.sum(weights * features) + bias
y = activation(x)

print(y)
```

    tensor([[0.1595]])



```python
# better to do matrix multiplication because it's optimized
torch.mm(weights, features)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-23-0459821846a9> in <module>
          1 # better to do matrix multiplication because it's optimized
    ----> 2 torch.mm(weights, features)
    

    RuntimeError: size mismatch, m1: [1 x 5], m2: [1 x 5] at ../aten/src/TH/generic/THTensorMath.cpp:961



```python

```

Since we are doing matrix multiplication, we need the matrices shapes to match.
We'll change the shape of weights for the mm to work.


```python
# weight.reshape(a, b) reshapes the data into a tensor of size (a, b)
# weight.resize_(a, b) returns the same tensor with a different shape.
# if 
# weight.view(a, b) returns a new tensor
print(weights.shape)
reshaped_weights = weights.view(5, 1)
y = activation(torch.mm(reshaped_weights, features) + bias)
print(y)

```

    torch.Size([1, 5])
    tensor([[0.6104, 0.4047, 0.3706, 0.7883, 0.2323],
            [0.5914, 0.5095, 0.4952, 0.6713, 0.4296],
            [0.5341, 0.7836, 0.8153, 0.2581, 0.9169],
            [0.5738, 0.6050, 0.6103, 0.5408, 0.6344],
            [0.6375, 0.2680, 0.2184, 0.8995, 0.0740]])



```python

```
