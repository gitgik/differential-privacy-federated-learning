# Differential Privacy & Federated Learning.
Curated notebooks on how to train neural networks using differential privacy and federated learning.

## What is Differential Privacy?
Differential Privacy is a set of techniques for
preventing a model from accidentally memorizing secrets present
in a training dataset during the learning process.

The key points under Differential Privacy are:

* Make a promise to a data subject that: You won’t be affected,
  adversely or otherwise, by allowing your data to be used in any analysis,
  no matter what studies, datasets or information sources, are available.
* Ensure that the model learning from sensitive data are only learning what they are
  supposed to learn without accidentally learning what they are not supposed to learn from their data


## What is Federated Learning
Instead of bringing data all to one place for training,
federated learning is done by bringing the model to the data.
This allows a data owner to maintain the only copy of their information.


## Intro Notebooks
Before you start learning about Differential Privacy, it's important to understand how tensors work.

### Learn about tensors:
- [Pytorch Tensors](intro-notebooks/pytorch_tensors.ipynb)
- [Reshaping Tensors](intro-noteboos/reshaping_tensors.ipynb)
- [Memory Sharing vs Copying](intro-notebooks/memory_sharing_vs_copying.ipynb)


### Creating simple neural networks
- [Single Layer Network](intro-notebooks/single_layer_network.ipynb)
- [Multi Layer Network](intro-notebooks/multilayer_network.ipynb)
- [Loading Image data](intro-notebooks/loading_image_data.ipynb)


### Transfer Learning
Most of the time you won't want to train a whole convolutional network yourself.
Modern ConvNets training on huge datasets like ImageNet take weeks on multiple GPUs.
[Transfer Learning](intro-notebooks/transfer_learning.ipynb) helps you solve this problem.


