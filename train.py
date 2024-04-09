from aepm import MultiModalGuesser
from h36m_dataset import H36M_Dataset

import os
import config
import datetime

import torch
from torch import nn
from tqdm import tqdm
from utils import sixteen_to_xyz, sixteen_to_angle
from torch.utils.tensorboard import SummaryWriter

@torch.no_grad()
def test(testloader, model):
    model.eval()
    MPJPE_mean_list = []
    MPJPE_best_list = []
    RMSE_mean_list = []
    RMSE_best_list = []

    for i, x in enumerate(testloader):
        x = x.to("cuda")
        x_masked = x.clone()
        x_masked[:, :, 2, :] = 0

        mean, var, sample = model(x_masked)
        extended_var = var[:, None, :, :, :] * torch.ones((x.shape[0], 10, 1, 1, 1)).cuda()
        std = torch.exp(extended_var/2)
        std_sample = sample.std(dim=1).mean(dim=(1, 2, 3))
        sample_prediction = sample * std / std_sample[:, None, None, None, None]
        x_hat = sample_prediction + mean[:, None]

        x_hat_mean = x_hat.mean(dim=1)
        
        gt = x.unsqueeze(1).repeat(1, 10, 1, 1, 1)
        losses = ((x_hat - gt) ** 2).mean(dim=(2, 3, 4))
        best_sample_index = losses.argmin(dim=1)
        x_hat_best = x_hat[torch.arange(x.shape[0]), best_sample_index]

        # calculate MPJPE and RMSE for mean and best sample
        gt_xyz = sixteen_to_xyz(x.view(-1, config.frame_num, 16, 3))[:, :, 3, :]
        x_hat_mean_xyz = sixteen_to_xyz(x_hat_mean.view(-1, config.frame_num, 16, 3))[:, :, 3, :]
        x_hat_best_xyz = sixteen_to_xyz(x_hat_best.view(-1, config.frame_num, 16, 3))[:, :, 3, :]
        mpjpe_mean = torch.mean(torch.norm(gt_xyz - x_hat_mean_xyz, dim=-1), dim=-1)
        mpjpe_best = torch.mean(torch.norm(gt_xyz - x_hat_best_xyz, dim=-1), dim=-1)

        gt_angle = sixteen_to_angle(x.view(-1, config.frame_num, 16, 3))[:, :, 2, :]
        x_hat_mean_angle = sixteen_to_angle(x_hat_mean.view(-1, config.frame_num, 16, 3))[:, :, 2, :]
        x_hat_best_angle = sixteen_to_angle(x_hat_best.view(-1, config.frame_num, 16, 3))[:, :, 2, :]
        rmse_mean = torch.sqrt(((gt_angle - x_hat_mean_angle) ** 2).mean(dim=-1))
        rmse_best = torch.sqrt(((gt_angle - x_hat_best_angle) ** 2).mean(dim=-1))

        MPJPE_mean_list.append(mpjpe_mean)
        MPJPE_best_list.append(mpjpe_best)
        RMSE_mean_list.append(rmse_mean)
        RMSE_best_list.append(rmse_best)

    MPJPE_mean = torch.cat(MPJPE_mean_list, dim=0).mean().item()
    MPJPE_best = torch.cat(MPJPE_best_list, dim=0).mean().item()
    RMSE_mean = torch.cat(RMSE_mean_list, dim=0).mean().item()
    RMSE_best = torch.cat(RMSE_best_list, dim=0).mean().item()

    model.train()
    return MPJPE_mean, MPJPE_best, RMSE_mean, RMSE_best


def train():
    model = MultiModalGuesser(num_frame=config.frame_num, num_joints=16, in_chans=3, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True).cuda()
    
    model.train()
    if config.load_pretrained:
        model.load_state_dict(torch.load(config.pretrained_path))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    dataset = H36M_Dataset("/root/autodl-tmp/datasets/h36m", "train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testset = H36M_Dataset("/root/autodl-tmp/datasets/h36m", "test")
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # create result folder and log file
    time_text = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_folder = os.path.join(config.base_path, "logs", time_text)
    writer_folder_path = os.path.join(config.base_path, "logs", time_text, "tensorboard")
    pth_save_path = os.path.join(config.base_path, "logs", time_text, "pth")
    os.makedirs(pth_save_path)
    writer = SummaryWriter(writer_folder_path)

    os.system(f"cp {config.base_path}/config.py {log_folder}/config.py")
    os.system(f"cp {config.base_path}/trainSTE_low.py {log_folder}/trainSTE_low.py")
    os.system(f"cp {config.base_path}/mixste.py {log_folder}/mixste.py")

    for epoch in range(config.epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=80)
        for i, x in pbar:
            x = x.to("cuda")
            x_masked = x.clone()
            x_masked[:, :, 2, :] = 0

            mean, var, sample = model(x_masked)
            x_hat = model.reparameterize(mean, var, sample)
            
            if epoch <= config.shift_epoch:
                gt = x.unsqueeze(1).repeat(1, 10, 1, 1, 1).view(-1, config.frame_num, 16, 3)
                x_hat = x_hat.view(-1, config.frame_num, 16, 3)
                loss = criterion(x_hat, gt)

            if epoch > config.shift_epoch:
                gt = x.unsqueeze(1).repeat(1, 10, 1, 1, 1)
                losses = ((x_hat - gt) ** 2).mean(dim=(2, 3, 4))
                best_sample_index = losses.argmin(dim=1)
                x_hat = x_hat[torch.arange(x.shape[0]), best_sample_index]
                loss_mse = criterion(x_hat, x)
                loss = loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch <= config.shift_epoch:
                    gt_xyz = sixteen_to_xyz(gt.view(-1, config.frame_num, 16, 3))[:, :, 3, :]
                else:
                    gt_xyz = sixteen_to_xyz(x.view(-1, config.frame_num, 16, 3))[:, :, 3, :]
                x_hat_xyz = sixteen_to_xyz(x_hat.view(-1, config.frame_num, 16, 3))[:, :, 3, :]
                mpjpe = torch.mean(torch.norm(gt_xyz - x_hat_xyz, dim=-1), dim=-1)

            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.6f} | MPJPE: {mpjpe.mean().item():.1f}")
            writer.add_scalar("Loss", loss.item(), epoch*len(dataloader)+i)
            writer.add_scalar("MPJPE", mpjpe.mean().item(), epoch*len(dataloader)+i)

        if epoch % config.test_interval == 0:

            MPJPE_mean, MPJPE_best, RMSE_mean, RMSE_best = test(testloader, model)
            writer.add_scalar("MPJPE_mean", MPJPE_mean, epoch)
            writer.add_scalar("MPJPE_best", MPJPE_best, epoch)
            writer.add_scalar("RMSE_mean", RMSE_mean, epoch)
            writer.add_scalar("RMSE_best", RMSE_best, epoch)

            print(f"Epoch {epoch} | MPJPE_mean: {MPJPE_mean:.6f} | MPJPE_best: {MPJPE_best:.6f} | RMSE_mean: {RMSE_mean:.6f} | RMSE_best: {RMSE_best:.6f}")

        torch.save(model.state_dict(), os.path.join(pth_save_path, f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    train()
