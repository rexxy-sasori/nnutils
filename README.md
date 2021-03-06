This repository is written to simplify the common neural network pipeline

In this repo, you will find: 1) code for training and evaluation of neural networks; 2) code for layerwise analysis of computational complexity

Installation:
```bash
pip install nn-pipeline
```

Example for analyzing complexity:
```python
from nnutils.cnn_complexity_analyzer.profile import profile_compute_layers

# model: torch.nn.Module
# inputs: dictionary type decribing input to the network
# Example: most CNN models on cifar10: inputs={'x': torch.randn(1,3,32,32)} 
# Keys in inputs needs to be the same as the parameter names for model's forward function

# profile results will look like:
# {
#    layer1.weight: ComputeProfile
#    ...
# }
# where the keys are the same as keys in model.state_dict()
# Attributes of ComputeProfile is found in nnutils.cnn_complexity_analyzer.utils.NNComputeModuleProfile
# Number of keys in profile results is the sum of CONV layers and FC layers in the network

profile_results, model_sparsity = profile_compute_layers(model, inputs=inputs, verbose=True)
```

Example for evaluating accuracy:
```python
from nnutils.training_pipeline import accuracy_evaluator

# model: torch.nn.Module
# device: 'cpu' or 'cuda' device to evalaute the network
# data_loader: dataset for the model
# criterion: loss function for the model

top1_acc, loss = accuracy_evaluator.eval(model, device, data_loader, criterion, print_acc=True)
```
