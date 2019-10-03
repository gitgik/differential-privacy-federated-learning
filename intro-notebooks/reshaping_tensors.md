
## Reshaping Pytorch Tensors


```python
import torch
```


```python
# tensor with rank two tensor with two axis,
# first axis: length = 3, second axis: length = 4
# elements of axis 1: arrays, elements of axis 2 = numbers
t = torch.tensor([
    [1, 2, 3, 4],
    [4, 4, 6, 8],
    [0, 4, 8, 12]
], dtype=torch.float32)
t.size()
```




    torch.Size([3, 4])




```python
print(t.shape)
len(t.shape) # the number of axis
# product of the tensor size shows the number of elements in the shape
torch.tensor(t.shape).prod() # 12 (scalar components)
```

    torch.Size([3, 4])





    tensor(12)




```python
# we can also use numel (number of elements)
```


```python
t.numel()
```




    12




```python
t.reshape(1, 12)

```




    tensor([[ 1.,  2.,  3.,  4.,  4.,  4.,  6.,  8.,  0.,  4.,  8., 12.]])




```python
t.reshape(4, 3)
```




    tensor([[ 1.,  2.,  3.],
            [ 4.,  4.,  4.],
            [ 6.,  8.,  0.],
            [ 4.,  8., 12.]])




```python
t.reshape(6, 2)
```




    tensor([[ 1.,  2.],
            [ 3.,  4.],
            [ 4.,  4.],
            [ 6.,  8.],
            [ 0.,  4.],
            [ 8., 12.]])



We can also change the tensor by squezing and unsqueezing allowing us to expand
or shrink a tensor.


```python
# Flatten a tensor: Change it into a lower rank tensor 
# (removing axis except from one-- creating a 1d array)
# Occurs when transitioning from a convolutional layer to a fully connected layer
print(t.reshape(1,12).squeeze())
print(t.reshape(1, 12).squeeze().unsqueeze(dim=0))
```

    tensor([ 1.,  2.,  3.,  4.,  4.,  4.,  6.,  8.,  0.,  4.,  8., 12.])
    tensor([[ 1.,  2.,  3.,  4.,  4.,  4.,  6.,  8.,  0.,  4.,  8., 12.]])



```python
def flatten(t):
    t = t.reshape(1, -1) # - 1 tells reshape to figure get the length(tensor)
    t = t.squeeze()
    return t

```


```python
flatten(t)
```




    tensor([ 1.,  2.,  3.,  4.,  4.,  4.,  6.,  8.,  0.,  4.,  8., 12.])



### Concatenate tensors

Use the `cat()` function, resulting in a tensor having a shape that depends on the shape of the two tensors.



```python
t1 = torch.tensor([
    [1,2],
    [3,4],
])
t2 = torch.tensor([
    [5, 6],
    [7, 8]
])
# combine the tensors on the row axis (axis-0)
torch.cat((t1, t2), dim=0)
```




    tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])




```python
# We can combine their column-axis (axis-1) like this
torch.cat((t1, t2), dim=1)
```




    tensor([[1, 2, 5, 6],
            [3, 4, 7, 8]])


