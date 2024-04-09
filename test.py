import torch
import numpy as np

from aepm import MultiModalGuesser
from h36m_dataset import H36M_Dataset
from utils import sixteen_to_xyz

model = MultiModalGuesser(num_frame=25, num_joints=16, in_chans=3, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True).cuda().eval()
model.load_state_dict(torch.load("/root/autodl-tmp/PTMM_clean/logs/20240321-005719/pth/epoch_123.pth"))
mse = torch.nn.MSELoss()

data_names = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting', 'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']

used_joint_indexes = np.array([0,1,2,6,7,11,12,13,14,15,16,17,18,24,25,26]).astype(np.int64)

@torch.no_grad()
def test():
    total_mpjpes = []
    total_mins = []
    for data_name in data_names:
        for num in ['_1', '_2']:
            target_data_path = '/root/autodl-tmp/datasets/h36m/S5/' + data_name + num + '.txt'
            target_data = torch.from_numpy(np.loadtxt(target_data_path, delimiter=',')).float().to("cuda")

            mpjpes = []
            mpjpes_min = []
            
            angles_gt = []
            angles_x_hat = []
            for i in range(10):
                angles_x_hat.append([])

            for i in range(0, len(target_data) - 25, 1):
                gt = target_data[i:i+25].to("cuda").reshape(25, 33, 3)
                gt = gt[:, 1:, :]
                gt = gt[:, used_joint_indexes, :]
                gt = gt.unsqueeze(0)

                gt_masked = gt.clone()
                gt_masked[:, :, 2, :] = 0

                mean, var, sample = model(gt_masked)
                extended_var = var[:, None, :, :, :] * torch.ones((gt.shape[0], 10, 1, 1, 1)).cuda()
                std = torch.exp(extended_var/2)
                std_sample = sample.std(dim=1).mean(dim=(1, 2, 3))
                sample_prediction = sample * std / std_sample[:, None, None, None, None]
                x_hat = sample_prediction + mean[:, None]
                x_hat[:, :, :, 2, 1:] = 0

                gt = gt.unsqueeze(1).repeat(1, 10, 1, 1, 1).view(-1, 25, 16, 3)
                x_hat = x_hat.view(-1, 25, 16, 3)

                xyz_hat = sixteen_to_xyz(x_hat)[:, :, 3, :]
                xyz_gt = sixteen_to_xyz(gt)[:, :, 3, :]

                mpjpe = torch.sqrt(torch.sum((xyz_hat - xyz_gt)**2, dim=-1)).mean()
                mpjpes.append(mpjpe.item())

                xyz_hat = xyz_hat.view(-1, 10, 25, 3)
                xyz_gt = xyz_gt.view(-1, 10, 25, 3)

                # calculate mpjpes that is the lowest among the 10, (1, 10)
                mpjpe_min = torch.sqrt(torch.sum((xyz_hat - xyz_gt)**2, dim=-1)).mean(dim=2).min(dim=1)[0].mean()
                mpjpes_min.append(mpjpe_min)

                # get the lowest 
                gt = gt.view(1, 10, 25, 16, 3)
                x_hat = x_hat.view(1, 10, 25, 16, 3)
                # losses = ((x_hat - gt) ** 2).mean(dim=(2, 3, 4))
                # best_sample_index = losses.argmin(dim=1)
                # x_hat = x_hat[torch.arange(x_hat.shape[0]), best_sample_index]
                gt = gt[:, 0, :, :, :]
                angles_gt.append(gt[:, :, 2, :].view(-1, 3)[-1, :].unsqueeze(0))
                for i in range(10):
                    # angles_x_hat[i].append(x_hat[:, i, :, 2, :].view(-1, 3))
                    # only append the last frame
                    angles_x_hat[i].append(x_hat[:, i, :, 2, :].view(-1, 3)[-1, :].unsqueeze(0))

            # angles_gt = torch.cat(angles_gt, dim=0)
            # angles_x_hat = torch.cat(angles_x_hat, dim=1)
            # turn list to numpy array
            angles_gt = torch.cat(angles_gt, dim=0)
            for i in range(10):
                angles_x_hat[i] = torch.cat(angles_x_hat[i], dim=0).unsqueeze(0)
            angles_x_hat = torch.cat(angles_x_hat, dim=0)
            # save the angles

            # save the output data
            output_data = angles_x_hat.cpu().numpy()
            np.save(f"/root/autodl-tmp/PTMM_clean/output_data_clean/{data_name}{num}.npy", output_data)

            import matplotlib.pyplot as plt
            plt.figure()
            # plt.plot(angles_x_hat[:1000, 0].cpu().numpy(), label="x_hat")
            for m in range(10):
                plt.plot(angles_x_hat[m][:1000, 0].cpu().numpy(), label=f"x_hat{m}", color='blue', alpha=0.5)
            plt.legend()
            # set ratio of x and y axis to 1:4
            plt.gca().set_aspect('auto', adjustable='box')
            plt.plot(angles_gt[:1000, 0].cpu().numpy(), label="gt", color='green')
            plt.savefig(f"/root/autodl-tmp/PTMM_clean/logs/pics/{data_name}{num}.png")

            # print(f"{data_name}{num} mpjpe: {sum(mpjpes)/len(mpjpes)}")
            print(f"{sum(mpjpes)/len(mpjpes)},{sum(mpjpes_min)/len(mpjpes_min)}")
            total_mpjpes.append(sum(mpjpes)/len(mpjpes))
            total_mins.append(sum(mpjpes_min)/len(mpjpes_min))
            
    print(f"mpjpe: {sum(total_mpjpes)/len(total_mpjpes)}")
    print(f"mpjpe_min: {sum(total_mins)/len(total_mins)}")


if __name__ == "__main__":
    test()