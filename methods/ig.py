import torch

class IntegratedGradients:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def explain(self, x, target_class=None, steps=50):
        if target_class is None:
            with torch.no_grad():
                target_class = self.model(x).argmax(dim=1).item()

        # Generate linear interpolation
        alphas = torch.linspace(0, 1, steps+1, device=x.device).view(-1, 1, 1, 1)
        # Baseline is black (zeros)
        scaled_inputs = x * alphas
        scaled_inputs.requires_grad = True

        preds = self.model(scaled_inputs)
        score = preds[:, target_class].sum()
        
        self.model.zero_grad()
        score.backward()
        
        grads = scaled_inputs.grad
        avg_grads = torch.mean(grads[:-1], dim=0)
        
        # IG = Input * AvgGrad
        ig = x * avg_grads
        
        # Collapse channels to create a heatmap (sum of absolute values)
        saliency = torch.sum(torch.abs(ig), dim=1, keepdim=True)
        return saliency