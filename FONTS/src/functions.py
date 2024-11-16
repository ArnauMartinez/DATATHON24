import torch

def compute_accuracy(output, target) -> float:

    print('OUTPUT', output,'TARGET', target)
    pred = torch.argmax(output, dim=1)
    
    target = torch.argmax(target, dim=1)
    
    # Compute the number of correct predictions
    correct = (pred == target).sum().item()
    
    # Return accuracy as a percentage
    accuracy = correct / target.size(0) * 100.0
    return accuracy