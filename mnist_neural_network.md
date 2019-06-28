

```python
import torch
from torchvision import datasets, transforms

# define a transform to normalize the data
# if the img has three channels, you should have three number for mean, 
# for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5. 
# If img is grey type that only one channel, mean should be [0.5], the normalize result is R * 0.5
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                               ])
# download and load the traning data
trainset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```


```python
# make an iterator for looping
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images[0].shape)
# NOTE: The batch size is the number of images we get in one iteration
```

    <class 'torch.Tensor'>
    torch.Size([1, 28, 28])
    torch.Size([64, 1, 28, 28])



```python
import matplotlib.pyplot as plt
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
```


![png](mnist_neural_network_files/mnist_neural_network_2_0.png)


Time to create a dense fully-connected network. 

Each unit in one layer is connected to the other in the next layer.
The input to each layer must be one-dimensional vector. But our images are 28*28 2D tensors, so we need to convert them to 1D vectors. Therefore:
* Convert/Flatten the batch of images of shape(64, 1, 28, 28) into (64, 28 * 28=784).
* For the output layer, we also need 10 output units for the 10 classes(digits)
* Also convert the network output into a probability distribution.


```python
flattened_images = images.view(64, 28 * 28)
```


```python
print(flattened_images.shape)
```

    torch.Size([64, 784])



```python
def activation(x):
    """Create a sigmoid activation function.
    Good for outputs that fall between 0 and 1. (probability)
    args x: a torch tensor.
    """
    return 1/(1 + torch.exp(-x))

def softmax(x):
    """Create a softmax activation function.
    Good for outputs that fall between 0 and 1. (probability)
    args x: a torch tensor.
    """
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)
```


```python
# flatten the images to shape(64, 784)
inputs = images.view(images.shape[0], -1)

# create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2
probabilities = softmax(out)
print(probabilities.shape)
print(probabilities.sum(dim=1))
```

    torch.Size([64, 10])
    tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000])


## Using the Torch nn to create networks


```python
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    """Use relu(Rectified linear unit) as the activation function.
    Networks tend to train a lot faster when using relu.
    For a network to approximate a non-linear function, the activation
    function must be non-linear.
    """
    def __init__(self):
        super().__init__()
        # inputs to hidden layer linear transformation
        self.hidden_layer1 = nn.Linear(784, 128) # 256 outputs
        self.hidden_layer2 = nn.Linear(128, 64)
        # output layer, 10 units one for each digit
        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        # hidden layer with sigmoid activation
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        return x
```


```python
model = Network()
model
```




    Network(
      (hidden_layer1): Linear(in_features=784, out_features=128, bias=True)
      (hidden_layer2): Linear(in_features=128, out_features=64, bias=True)
      (output): Linear(in_features=64, out_features=10, bias=True)
    )



## Training our network



```python
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10)
                     )

# define the loss
criterion = nn.CrossEntropyLoss()

# Prepare data
images, labels = next(iter(trainloader))

# flatten images
images = images.view(images.shape[0], -1)

# forward pass, get the logits
logits = model(images)
# calculate the loss with the logits and the labels
loss = criterion(logits, labels)
print(loss)
```

    tensor(2.3058, grad_fn=<NllLossBackward>)


It's more convenient to build model with a log-softmax output using `nn.LogSoftmax`
We can get actual probabilities by taking the exponential torch.exp(output).
We'll also use the negative log likelihood loss, `nn.NLLLoss`


```python
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1),
                     )
criterion = nn.NLLLoss()

logits = model(images)
loss = criterion(logits, labels)
print(loss)

```

    tensor(2.3025, grad_fn=<NllLossBackward>)


## USing Autograd to perform backpropagation

After calculating loss, we perform backpropagation. Enter `autograd`

We use it to calculate the gradients of all our parameters with respect to the loss we got. Autograd goes backwards through the tensor operations, calculating gradients along the way. 
* Set `requires_grad=True` on a tensor when creating the tensor.


```python
x = torch.randn(2,2, requires_grad=True)
y = x** 2
z = y.mean()
z.backward()
print(x.grad)
```

    tensor([[-0.4997, -0.1425],
            [-0.8944,  0.0633]])
    tensor([[-0.4997, -0.1425],
            [-0.8944,  0.0633]], grad_fn=<DivBackward0>)



```python
# Back to the model we created
```


```python
print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)
```

    Before backward pass: 
     None
    After backward pass: 
     tensor([[-0.0029, -0.0029, -0.0029,  ..., -0.0029, -0.0029, -0.0029],
            [-0.0028, -0.0028, -0.0028,  ..., -0.0028, -0.0028, -0.0028],
            [-0.0006, -0.0006, -0.0006,  ..., -0.0006, -0.0006, -0.0006],
            ...,
            [-0.0011, -0.0011, -0.0011,  ..., -0.0011, -0.0011, -0.0011],
            [-0.0036, -0.0036, -0.0036,  ..., -0.0036, -0.0036, -0.0036],
            [-0.0026, -0.0026, -0.0026,  ..., -0.0026, -0.0026, -0.0026]])


We also need an optimizer that'll update weights with the gradients from the backward pass.
From Pytorch's `optim` package, we can use stochastic gradient descenc with `optim.SGD`




```python
from torch import optim
# pass in the parameter to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)
```


```python
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1),
                     )
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten Images into 784 long vector for the input layer
        images = images.view(images.shape[0], -1)
        
        # clear gradients because they accumulate
        optimizer.zero_grad()
        # forward pass
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f'Training loss: {running_loss/len(trainloader)}')
```

    Training loss: 1.0614541531689385
    Training loss: 0.3895804712147728
    Training loss: 0.3246205799631091
    Training loss: 0.29300587304206543
    Training loss: 0.26864237868900237



```python
# create a helper to view the probability distribution
import matplotlib.pyplot as plt
import numpy as np

def view_classify(img, ps):
    """Function for viewing an image and it's predicted classes."""
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))

```


```python

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)

with torch.no_grad():
    logits = model.forward(img)
    
    ps = F.softmax(logits, dim=1)
    print(ps)
    view_classify(img.view(1, 28, 28), ps)
```

    tensor([[1.2855e-02, 5.0043e-05, 7.8326e-04, 5.7256e-03, 4.5138e-03, 9.6925e-01,
             1.3272e-03, 1.2127e-03, 5.4744e-04, 3.7324e-03]])



![png](mnist_neural_network_files/mnist_neural_network_24_1.png)

