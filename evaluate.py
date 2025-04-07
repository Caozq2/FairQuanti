import torch
import pandas as pd
from pycm import ConfusionMatrix

class bm:
    def __init__(self, df):
        self._df = df

    def P(self, **kwargs):
        """
        Declares the random variables from the set `kwargs`.
        """
        self._variables = kwargs
        return self

    def given(self, **kwargs):
        """
        Calculates the probability on a finite set of samples with `kwargs` in the
        conditioning set. 
        """
        self._given = kwargs
        
        # Here's where the magic happens
        prior = True
        posterior = True
        
        for k in self._variables:
            if type(self._variables[k]) == type(lambda x:x):
                posterior = posterior & (self._df[k].apply(self._variables[k]))
            else:
                posterior = posterior & (self._df[k] == self._variables[k])

        
        for k in self._given:
            if type(self._given[k]) == type(lambda x:x):
                prior = prior & (self._df[k].apply(self._given[k]))
                posterior = posterior & (self._df[k].apply(self._given[k]))
            else:
                prior = prior & (self._df[k] == self._given[k])
                posterior = posterior & (self._df[k] == self._given[k])
        return posterior.sum()/prior.sum()



def get_metrics(results, threshold, fraction,dataset='compas'):
    "Create the metrics from an output df."

    # Calculate biases after training
    dem_parity = abs(
        bm(results).P(pred=lambda x: x > threshold).given(race=0)
        - bm(results).P(pred=lambda x: x > threshold).given(
            race=1))

    eq_op = abs(
        bm(results).P(pred=lambda x: x > threshold).given(race=0, compas=True)
        - bm(results).P(pred=lambda x: x > threshold).given(race=1, compas=True))

    epsilon = 1e-10

    dem_parity_ratio = abs(
        bm(results).P(pred=lambda x: x > threshold).given(race=0)
        / (bm(results).P(pred=lambda x: x > threshold).given(
            race=1) + epsilon))

    cm = ConfusionMatrix(actual_vector=(results['true'] == True).values,
                         predict_vector=(results['pred'] > threshold).values)

    result = {"DP": dem_parity,
                "EO": eq_op,
                "DP ratio": dem_parity_ratio,
                "acc": cm.Overall_ACC,
                "acc_ci_min": cm.CI95[0],
                "acc_ci_max": cm.CI95[1],
                "f1": cm.F1_Macro,
                "adversarial_fraction": fraction
                }

    return result

# 评估模型性能
def evaluate_model(model, test_loader, device, criterion, threshold=0.5):
    model.to(device)
    model.eval()
    test_losses = []
    test_results = []
    with torch.no_grad():
        for x_test, y_test, ytrue, s_true in test_loader:
            x_test, y_test, s_true = x_test.to(device), y_test.to(device), s_true.to(device)
            x_new = torch.cat([x_test, s_true], dim=1)
            yhat = model(x_new)
            test_loss = criterion(yhat, y_test)
            test_losses.append(test_loss.item())
            test_results.append({"y_hat": yhat, "y_true": ytrue, "y_compas": y_test, "s": s_true})
    
    results = torch.cat([r['y_hat'] for r in test_results])
    outcome = torch.cat([r['y_true'] for r in test_results])
    compas = torch.cat([r['y_compas'] for r in test_results])
    protected_results = torch.cat([r['s'] for r in test_results])
    
    df = pd.DataFrame(data=results.cpu().numpy(), columns=['pred'])
    df['true'] = outcome.cpu().numpy()
    df['compas'] = compas.cpu().numpy()
    df['race'] = protected_results.cpu().numpy()[:, 0]
    return df