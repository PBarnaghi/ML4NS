import typing # for type hints
import torch
import torch.nn as nn
import tqdm

@torch.no_grad() # no gradients to calculate
def predict(
    model:nn.Module, 
    test_loader:torch.utils.data.DataLoader,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    '''
    This function will iterate over the :code:`test_loader`
    and return the outputs of the model
    applied to the data.

    Arguments
    ---------

    - model: nn.Module:
        The model used to make predictions.

    - test_loader: torch.utils.data.DataLoader:
        The data to predict on.
    
    Returns
    ---------

    - outputs: torch.Tensor:
        The outputs of the model
    
    - labels: torch.Tensor:
        The ground truth labels collected
        from :code:`test_loader`.

    '''
    
    # lists to contain the output data
    targets = []
    outputs = []

    # adding model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # iterating over the test_loader with a progress bar
    for input, target in tqdm.tqdm(test_loader, desc='Predicting'):
        # ==== push data to GPU if available ====
        input = input.to(device)
        # ==== forward pass ====
        output = model(input)
        # ==== saving outputs and labels ====
        outputs.append(output.cpu())
        targets.append(target) # target was never pushed to GPU so remains on cpu
    
    # turning outputs into torch tensors instead of lists
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)

    model.to('cpu') # return the model to the CPU

    return outputs, targets