import torch
import torch.nn as nn

def update_mask(mask, x, model, mask_opt, simp_weight, scale_factor, device='cuda'):

    im_size = 224
    mask = mask.requires_grad_(True)
    model.eval()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    batch_D_d, y, pred_fb_d = x

    batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(device), ups(mask).to(device), pred_fb_d.to(device)

    # get random background color to replace masked pixels
    background_means = torch.ones((1, 3))*torch.Tensor([0.527, 0.447, 0.403])
    background_std = torch.ones((1, 3))*torch.Tensor([0.05, 0.05, 0.05])
    avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(device)
    # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)

    pred_fs_d = sm(model(batch_D_d))
    pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

    # calculate loss by comparing the two models (t1) and the two datasets (t2)
    t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
    sim_heur = torch.linalg.vector_norm(batch_mask, 1)/(im_size*im_size)
    loss = ((simp_weight*sim_heur + t2))

    mask_opt.zero_grad()
    loss.backward()
    mask_opt.step()

    with torch.no_grad():

        mask.copy_(mask.clamp(max=1, min=0))
        mask_l0_norm = (torch.linalg.vector_norm(batch_mask.flatten(), 0)/(batch_mask.shape[2]*batch_mask.shape[3])).detach().cpu()

    metrics = torch.Tensor([loss.item(), torch.sum(sim_heur).item(), torch.sum(t2).item(), mask_l0_norm.item()])

    return metrics

def distill_singular(mask, model, x, mask_opt, mask_scale):

    num_rounding_steps = 1
    # simp_weight = [1- r*(0.9/num_rounding_steps) for r in range(num_rounding_steps)]
    simp_weight = 1

    mask_converged = False
    prev_loss, prev_prev_loss = float('inf'), float('inf')

    while (not mask_converged):

        mask_metrics = update_mask(mask, x, model, mask_opt, simp_weight, mask_scale)
        mask_loss = mask_metrics[0]
        mask_converged = (mask_loss > 0.998*prev_prev_loss) and (mask_loss < 1.002*prev_prev_loss)
        
        prev_prev_loss = prev_loss
        prev_loss = mask_loss

    return mask

class DiET:
    def __init__(self, model, im_size=224, mask_scale=8, device='cuda'):
        self.model = model
        self.im_size = im_size
        self.mask_scale = mask_scale
        self.device = device

    def explain(self, x, y):
        sm = torch.nn.Softmax(dim=1)
        pred_f = sm(self.model(x))

        mask = torch.ones((1, 1, self.im_size//self.mask_scale, self.im_size//self.mask_scale))
        mask = mask.requires_grad_(True)
        mask_opt = torch.optim.SGD([mask], lr=300)
        mask_opt.zero_grad()

        mask = distill_singular(mask, self.model, (x, y, pred_f), mask_opt, self.mask_scale)
        ups = torch.nn.Upsample(scale_factor=self.mask_scale, mode='bilinear')
        return ups(mask)