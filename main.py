import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.transform_dataset import transform_dataset, transform_dataset_credit, transform_dataset_census
from sklearn import preprocessing
from evaluate import get_metrics, evaluate_model
from models import Net, Net_CENSUS, NetPlus_
from skopt import gp_minimize
from skopt.space import Real
from pycm import ConfusionMatrix



class DataClass:
    def __init__(self, df, dataset):
        if dataset == 'compas':
            df_binary, Y, S, Y_true = transform_dataset(df)
            Y = Y.to_numpy()
            self.l_tensor = torch.tensor(Y_true.to_numpy().reshape(-1, 1).astype(np.float32))
            self.threshold = 4
        elif dataset == 'credit':
            df_binary, Y, S, Y_true = transform_dataset_credit(df)
            self.l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
            self.threshold = 0.5
        else: 
            df_binary, Y, S, Y_true = transform_dataset_census(df)
            self.l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
            self.threshold = 0.5
        
        self.x_tensor = torch.tensor(df_binary.to_numpy().astype(np.float32))
        self.y_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        self.s_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray().astype(np.float32))
        self.dataset = TensorDataset(self.x_tensor, self.y_tensor, self.l_tensor, self.s_tensor)
        
        base_size = len(self.dataset) // 10
        split = [7 * base_size, 1 * base_size, len(self.dataset) - 8 * base_size]
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, split)



def get_layer_outputs(model, x):
    outputs = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            outputs[name] = output.detach()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        model(x)
    
    for h in hooks:
        h.remove()
    
    return outputs

def compute_bias_neuron_index(model, x_batch, s_batch, device):
    model.eval()
    with torch.no_grad():
        x = x_batch.to(device, dtype=torch.float32)
        s = s_batch.to(device, dtype=torch.float32)
        flipped_s = torch.zeros_like(s)
        flipped_s[:, 0] = s[:, 1]
        flipped_s[:, 1] = s[:, 0]
        

        original_input = torch.cat([x, s], dim=1)  
        flipped_input = torch.cat([x, flipped_s], dim=1)
        
        original_outputs = get_layer_outputs(model, original_input)
        flipped_outputs = get_layer_outputs(model, flipped_input)
        
        layer_diffs = {}
        for layer_name, module in model.named_modules():
            if layer_name in original_outputs and isinstance(module, nn.Linear):
                orig_act = original_outputs[layer_name]  
                flip_act = flipped_outputs[layer_name]   
                diff = torch.mean(torch.abs(orig_act - flip_act), dim=0)  
                
                if layer_name == 'fc1':
                    layer_diffs[layer_name] = torch.tanh(diff).unsqueeze(0).repeat(module.weight.shape[0], 1)
                else:
                    layer_diffs[layer_name] = torch.tanh(diff)
            
        return layer_diffs

def find_initial_threshold(layer_diffs):
    all_diffs = torch.cat([d.flatten() for d in layer_diffs.values()]).cpu().numpy()
    sorted_diffs = np.sort(all_diffs)
    n = len(sorted_diffs)
    
    for i in range(n):
        t = sorted_diffs[i]
        bnr = (n - i) / n
        if bnr <= t:
            return t
    return 0.5

def apply_threshold_mask(model, layer_diffs, threshold, device, input_shape, s_shape):
    masked_model = type(model)(input_shape=input_shape, s_shape=s_shape).to(device)
    masked_model.load_state_dict(model.state_dict())  
    masks = {name: (diffs > threshold).to(device) for name, diffs in layer_diffs.items()}
    return masked_model, masks

def apply_masks(model, masks):
    hooks = []
    
    def mask_hook_fn(name):
        def hook(module, input, output):
            if name in masks:
                mask = masks[name]
                zeroed_output = output.clone()
                if name == 'fc1' and mask.dim() > 1:
                    mask = mask.any(dim=1)
                zeroed_output[:, mask] = 0
                return zeroed_output
            return output
        return hook
    
    for name, module in model.named_modules():
        if name in masks:
            hooks.append(module.register_forward_hook(mask_hook_fn(name)))
    
    return hooks

def optimize_threshold(model, test_loader, device, layer_diffs, initial_threshold, dataset_name, threshold, input_shape, s_shape):
    criterion = nn.MSELoss().to(device)

    def objective(t):
        t = t[0]
        masked_model, masks = apply_threshold_mask(model, layer_diffs, t, device, input_shape, s_shape)
        
        hooks = apply_masks(masked_model, masks)
        
        df = evaluate_model(masked_model, test_loader, device, criterion, threshold=threshold)
        metrics = get_metrics(df, threshold=threshold, fraction=0.0, dataset=dataset_name)
        
        dp_ratio = metrics.get('DP ratio', 0)
        dp = metrics.get('DP', 0)
        eo = metrics.get('EO', 0)
        
        dp_ratio_score = -abs(dp_ratio - 1)   
        dp_score = -dp
        eo_score = -eo
        
        fairness_score = (dp_ratio_score + dp_score + eo_score) / 3        
        accuracy_score = metrics.get('acc', 0)
        return -(0.7 * accuracy_score + 0.3 * fairness_score)
    
    res = gp_minimize(objective, [(0.0, 1.0)], n_calls=15, x0=[initial_threshold], random_state=42)
    return res.x[0]

def quantize_mixed_precision(model, layer_diffs, threshold, device, bits=8):
    input_features = model.fc1.weight.shape[1]
    quantized_model = type(model)(input_shape=input_features-2, s_shape=2).to(device)
    quantized_model.load_state_dict(model.state_dict())
    
    total_weights = 0
    low_precision_weights = 0
    
    for name, module in quantized_model.named_modules():
        if name in layer_diffs and hasattr(module, 'weight'):
            bias_mask = (layer_diffs[name] > threshold)
            with torch.no_grad():
                if name == 'fc1':
                    for i in range(module.weight.shape[0]):
                        total_weights += module.weight[i].numel()
                        if bias_mask[i].any():
                            module.weight[i] = quantize_tensor(module.weight[i], bits=2)
                            low_precision_weights += module.weight[i].numel()
                        else:
                            module.weight[i] = quantize_tensor(module.weight[i], bits=bits)
                else:
                    biased_indices = torch.where(bias_mask)[0]
                    normal_indices = torch.where(~bias_mask)[0]
                    total_weights += module.weight.numel()
                    if len(biased_indices) > 0:
                        module.weight[:, biased_indices] = quantize_tensor(
                            module.weight[:, biased_indices].clone(), bits=2)
                        low_precision_weights += module.weight[:, biased_indices].numel()
                    if len(normal_indices) > 0:
                        module.weight[:, normal_indices] = quantize_tensor(
                            module.weight[:, normal_indices].clone(), bits=bits)
    
    print(f"Quantization: {low_precision_weights}/{total_weights} weights at 2-bit precision")
    return quantized_model

def quantize_tensor(tensor, bits):
    qmin, qmax = 0, 2**bits - 1
    tensor_min, tensor_max = tensor.min(), tensor.max()
    scale = (tensor_max - tensor_min) / (qmax - qmin) if tensor_max != tensor_min else 1.0
    zero_point = tensor_min
    q_tensor = ((tensor - zero_point) / scale).round().clamp(qmin, qmax)
    return q_tensor * scale + zero_point

def fair_quantization(test_loader, model, optimizer, device, threshold, dataset, bits, input_shape, s_shape):
    for x_batch, y_batch, ytrue_batch, s_batch in test_loader:
        break
    
    layer_diffs = compute_bias_neuron_index(model, x_batch, s_batch, device)
    initial_threshold = find_initial_threshold(layer_diffs)
    optimal_threshold = optimize_threshold(model, test_loader, device, layer_diffs, initial_threshold, dataset, threshold, input_shape, s_shape)
    quantized_model = quantize_mixed_precision(model, layer_diffs, optimal_threshold, device, bits=bits)
    return quantized_model


def run(dataset, inputpath, outputpath, BATCH_SIZE, epochs=50, bits=8, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if dataset == 'credit':
        inputpath = inputpath or './data/Credit/german_credit'
        threshold = 0.5
    elif dataset == 'census':
        inputpath = inputpath or './data/Census/adult'
        threshold = 0.5
    elif dataset == 'compas':
        inputpath = inputpath or './data/COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv'
        threshold = 4
    
    file_name = f"{dataset}_{int(time.time())}"
    output_file = os.path.join(outputpath, file_name)
    print(f"Output will be saved to: {output_file}")
    
    df = pd.read_csv(inputpath, sep=' ' if dataset == 'credit' else ',')
    data_class = DataClass(df, dataset)
    
    input_shape = data_class.x_tensor.shape[1]
    s_shape = data_class.s_tensor.shape[1]
   
    train_loader = DataLoader(dataset=data_class.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=data_class.val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=data_class.test_dataset, batch_size=BATCH_SIZE)
        
    if dataset == 'credit':
        model = Net(input_shape=input_shape, s_shape=s_shape).to(device, dtype=torch.float32)
    elif dataset == 'census':
        model = Net_CENSUS(input_shape=input_shape, s_shape=s_shape).to(device, dtype=torch.float32)
    elif dataset == 'compas':
        model = NetPlus_(input_shape=input_shape, s_shape=s_shape).to(device, dtype=torch.float32)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    model.load_state_dict(torch.load('/data/Newdisk/caozhiqi/code/FairNeuron-main/FN/FairQuanti/results/compas_1743991690_trained_model.pth'))   
    
    print("\nApplying fairness-aware quantization...")

    quantized_model = fair_quantization(
        test_loader=test_loader, 
        model=model, 
        optimizer=optimizer, 
        device=device, 
        threshold=threshold, 
        dataset=dataset,
        bits=bits,
        input_shape=input_shape,
        s_shape=s_shape
    )

    df = evaluate_model(quantized_model, test_loader, device, nn.BCEWithLogitsLoss(), threshold=threshold)
    result = get_metrics(df, threshold=threshold, fraction=0.0, dataset=dataset)
    print("Final Metrics:", result)
    
    os.makedirs(outputpath, exist_ok=True)
    pd.DataFrame([result]).to_csv(f"{output_file}_results.csv", index=False)
    torch.save(model.state_dict(), f"{output_file}_quantized_model.pt")

    print(f"Experiment completed. Results saved to {output_file}")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fairness-aware neural network quantization')
    parser.add_argument('--dataset', choices={'compas', 'census', 'credit'}, default='compas')
    parser.add_argument('--batch-size', type=int, default=128, dest='batchsize')
    parser.add_argument('--bits', type=int, default=8, dest='bits')
    parser.add_argument('--input-path', default=None, dest='inputpath')
    parser.add_argument('--save-dir', default='./results', dest='outputpath')
    
    args = parser.parse_args()
    run(args.dataset, args.inputpath, args.outputpath, args.batchsize, args.bits)