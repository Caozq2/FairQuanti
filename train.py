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

# 数据类
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
        else:  # census
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


# 添加训练函数
def train_model(model, train_loader, val_loader, device, criterion, optimizer, epochs=50, patience=10):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for x_batch, y_batch, _, s_batch in train_loader:
            x_batch, y_batch, s_batch = x_batch.to(device), y_batch.to(device), s_batch.to(device)
            
            # 拼接x和s作为输入
            input_tensor = torch.cat([x_batch, s_batch], dim=1)
            
            optimizer.zero_grad()
            outputs = model(input_tensor)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val, _, s_val in val_loader:
                x_val, y_val, s_val = x_val.to(device), y_val.to(device), s_val.to(device)
                
                # 拼接x和s作为输入
                input_tensor = torch.cat([x_val, s_val], dim=1)
                
                outputs = model(input_tensor)
                batch_loss = criterion(outputs, y_val)
                val_loss += batch_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                model.load_state_dict(best_model_state)
                break
    
    if patience_counter < patience:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses



def run(dataset, inputpath, outputpath, BATCH_SIZE, epochs=50, patience=10):
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
    
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        patience=patience
    )

    df_trained = evaluate_model(model, test_loader, device, criterion, threshold=threshold)
    trained_metrics = get_metrics(df_trained, threshold=threshold, fraction=0.0, dataset=dataset)
    print("Trained Metrics:", trained_metrics)

    os.makedirs(outputpath, exist_ok=True)
    torch.save(model.state_dict(), f"{output_file}_trained_model.pth")
    print(f"Trained model saved to {output_file}_trained_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fairness-aware neural network quantization')
    parser.add_argument('--dataset', choices={'compas', 'census', 'credit'}, default='compas')
    parser.add_argument('--batch-size', type=int, default=128, dest='batchsize')
    parser.add_argument('--input-path', default=None, dest='inputpath')
    parser.add_argument('--save-dir', default='./results', dest='outputpath')
    
    args = parser.parse_args()
    run(args.dataset, args.inputpath, args.outputpath, args.batchsize)