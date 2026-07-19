from __future__ import print_function
import numpy as np
import numpy.ma as ma
import time
import sys
import shutil
import yaml
from argparse import ArgumentParser
import os
from sklearn import metrics
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models.GaiaAgent_VLMTool import GaiaAgent_VLMTool
from dataloader import Dataset
from augmentations import get_validation_augmentations, get_training_augmentations
from losses import choose_criterion2d, choose_criterion3d
from optim import set_optimizer, set_scheduler
from sklearn.metrics import confusion_matrix


def get_args():
    parser = ArgumentParser(description = "Hyperparameters", add_help = True)
    parser.add_argument('-c', '--config-name', type = str, help = 'YAML Config name', dest = 'CONFIG', default = 'config')
    parser.add_argument('-nw', '--num-workers', type = str, help = 'Number of workers', dest = 'num_workers', default = 2)
    parser.add_argument('-v', '--verbose', type = bool, help = 'Verbose validation metrics', dest = 'verbose', default = False)
    return parser.parse_args()

# to calculate rmse
def metric_mse(inputs, targets, mask, exclude_zeros=False):
    if exclude_zeros:
        mask_ = (mask == 0)  

        # 应用掩码
        inputs = ma.masked_array(inputs, mask=mask_)
        targets = ma.masked_array(targets, mask=mask_)

        # 计算误差
        loss = (inputs - targets) ** 2
        n_pixels = np.count_nonzero(~mask_)  # 计算非 0 类的像素数量
        
        if n_pixels == 0:  # 避免除零错误
            return np.nan  # 或者返回一个默认值

        return np.sum(loss) / n_pixels
    else:
        loss = (inputs - targets) ** 2
        return np.mean(loss)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

args = get_args()


gpu_ids = [1]  
device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")


manual_seed = 18
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

config_name = args.CONFIG
config_path = './config/'+config_name
default_dst_dir = "./results/"
out_file = default_dst_dir + config_name + '/'
os.makedirs(out_file, exist_ok=True)

# Load the configuration params of the experiment
full_config_path = config_path + ".yaml"
print(f"Loading experiment {full_config_path}")
with open(full_config_path, "r") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)

print(f"Logs and/or checkpoints will be stored on {out_file}")
shutil.copyfile(full_config_path, out_file+'config.yaml')
print("Config file correctly saved!")

stats_file = open(out_file + 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv), file=stats_file)
print(' '.join(sys.argv))

print(exp_config)
print(exp_config, file=stats_file)

x_train_dir = exp_config['data']['train']['path']
x_valid_dir = exp_config['data']['val']['path']
x_test_dir = exp_config['data']['test']['path']

batch_size = exp_config['data']['train']['batch_size']

lweight2d, lweight3d = exp_config['model']['loss_weights']
weights2d = exp_config['model']['2d_loss_weights']

augmentation = exp_config['data']['augmentations']
min_scale = exp_config['data']['min_value']
max_scale = exp_config['data']['max_value']

mean = exp_config['data']['mean']
std = exp_config['data']['std']

if augmentation:
    train_transform = get_training_augmentations(m = mean, s = std)
else:
  train_transform = get_validation_augmentations(m = mean, s = std)

valid_transform = get_validation_augmentations(m = mean, s = std)

train_dataset = Dataset(x_train_dir,
                        text_dir=exp_config['data']['train']['text_path'],
                        augmentation=train_transform,
                        mode="train")

valid_dataset = Dataset(x_valid_dir,
                        augmentation=valid_transform,
                        mode="val")

test_dataset = Dataset(x_test_dir,
                       augmentation=valid_transform,
                       mode="test")


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

name_3dloss = exp_config['model']['3d_loss']
exclude_zeros = exp_config['model']['exclude_zeros'] 
criterion3d = choose_criterion3d(name = name_3dloss)


weight = torch.tensor(weights2d, device=device, dtype=torch.float32)
criterion2d = choose_criterion2d(
    name=exp_config['model']['2d_loss'],
    class_weights=weight,
    class_ignored=exp_config['model']['class_ignored'],
    alpha=exp_config['model'].get('focal_alpha', 0.5),
    gamma=exp_config['model'].get('focal_gamma', 2),
)


nepochs = exp_config['optim']['num_epochs']
lr = exp_config['optim']['lr']

model = exp_config['model']['model']
classes = exp_config['model']['num_classes']

if model != 'GaiaAgent_VLMTool':
    raise ValueError(f"Only GaiaAgent_VLMTool is supported, got {model}")
net = GaiaAgent_VLMTool().to(device)

print('Model selected: ', model)


if torch.cuda.device_count() > 1:
    print(f"Using GPUs {gpu_ids} for training.")
    net = torch.nn.DataParallel(net, device_ids=gpu_ids)


optimizer = set_optimizer(exp_config['optim'], net)
print('Optimizer selected: ', exp_config['optim']['optim_type'])
lr_adjust = set_scheduler(exp_config['optim'], optimizer)
print('Scheduler selected: ', exp_config['optim']['lr_schedule_type'])
res_cp = exp_config['model']['restore_checkpoints']
ckpt_path = out_file + f'{res_cp}bestnet.pth'

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint, strict=False)
    print(f'Checkpoints successfully loaded from {ckpt_path} to {device}!')
else:
    print(f'No checkpoints found at {ckpt_path}')


start = time.time()

best2dmetric = 0
best3dmetric = 1000000

net.train()

for epoch in range(1, nepochs):
  tot_2d_loss = 0
  tot_3d_loss = 0
  tot_con_loss = 0

  for param_group in optimizer.param_groups:
    print("Epoch: %s" % epoch, " - Learning rate: ", param_group['lr'])

  for text, t1, t2, mask2d, mask3d in tqdm(train_loader, desc="Training"):
    # print(f"mask3d shape: {mask3d.shape}")
    # print(f"mask2d shape: {mask2d.shape}")

    t1 = t1.to(device)
    t2 = t2.to(device)
    # print(text)
    # text = text.to(device)

    mask3d = mask3d.to(device).float()
    loss, out2d, out3d = net(t1, t2, text, train_mode=True)

    if args.verbose:
        print()
        print('MASK 3D: ', torch.min(mask3d).item(), torch.max(mask3d).item())

    mask3d = 2*(mask3d - min_scale)/(max_scale - min_scale)-1
    # mask2d_1 = mask2d.to(device).float().unsqueeze(1)

    if args.verbose:
        print('MASK 3D NORM: ', torch.min(mask3d).item(), torch.max(mask3d).item())
        print('OUT 3D: ', torch.min(out3d).item(), torch.max(out3d).item())

    loss2d = criterion2d(out2d, mask2d.long().to(out2d.device))

    loss3d = criterion3d(out3d.squeeze(dim=1), mask3d) #, exclude_zeros = exclude_zeros)
    loss_con = loss

    loss = lweight2d*loss2d + lweight3d*loss3d + 0.1*loss_con #sommo le loss
    # loss = lweight2d*loss2d + lweight3d*loss3d +loss_t#sommo le loss

    optimizer.zero_grad()
    loss.backward() #backward delle loss
    optimizer.step()

    tot_2d_loss += loss2d.detach().cpu().numpy()*batch_size
    tot_3d_loss += loss3d.detach().cpu().numpy()*batch_size
    tot_con_loss += loss_con.detach().cpu().numpy()*batch_size

  epoch_2d_loss = tot_2d_loss/len(train_dataset)
  epoch_3d_loss = tot_3d_loss/len(train_dataset)
  epoch_con_loss = tot_con_loss/len(train_dataset)

  epoch_loss = lweight2d*epoch_2d_loss + lweight3d*epoch_3d_loss + 0.1*epoch_con_loss


  lr_adjust.step()

  print(f"Training loss: {epoch_loss},\t2D Loss: {epoch_2d_loss}, \t3D Loss: {epoch_3d_loss}, \tContrastive Loss: {epoch_con_loss}")

  with torch.no_grad():
    net.eval()

    TN = 0
    FP = 0
    FN = 0
    TP = 0
    mean_mae = 0
    rmse1 = 0
    rmse2 = 0

    for t1, t2, mask2d, mask3d in tqdm(valid_loader, desc="Validation"):

      t1 = t1.to(device)
      t2 = t2.to(device)

      out2d, out3d = net(t1, t2, train_mode=False)
      out2d = out2d.detach().argmax(dim=1).cpu().numpy()
      out3d = out3d.detach().cpu().numpy()
      out3d = ((out3d.ravel()+1)/2)*(max_scale-min_scale)+min_scale

      # 确保 `confusion_matrix()` 输出固定形状
      num_classes = 7
      cm = confusion_matrix(mask2d.ravel(), out2d.ravel(), labels=np.arange(num_classes))
      # print(f"Confusion matrix shape: {cm.shape}")

      # 计算 TP, FP, FN, TN
      tp = np.diag(cm)
      fp = cm.sum(axis=0) - tp
      fn = cm.sum(axis=1) - tp
      tn = cm.sum() - (tp + fp + fn)

      mean_ae = metrics.mean_absolute_error(mask3d.ravel(), out3d.ravel())
      s_rmse1 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros = False)
      s_rmse2 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros = True)
      max_error = metrics.max_error(mask3d.ravel(), out3d.ravel())
      mask_max = np.abs(mask3d.cpu().numpy()).max()

      mean_mae += mean_ae
      rmse1 += s_rmse1
      rmse2 += s_rmse2
      TN += tn
      FP += fp
      FN += fn
      TP += tp

    mean_mae = mean_mae/len(valid_loader)
    mIoU = TP/(TP+FN+FP)
    mIoU = mIoU.mean()
    mean_f1 = 2*TP/(2*TP+FP+FN)
    mean_f1 = mean_f1.mean()  # 计算所有类别的平均 F1
    RMSE1 = np.sqrt(rmse1/len(valid_loader))
    RMSE2 = np.sqrt(rmse2/len(valid_loader))

    print(f'Validation metrics - 2D: F1 Score -> {mean_f1*100} %; mIoU -> {mIoU*100} %; 3D: MAE -> {mean_mae} m; RMSE -> {RMSE1} m; cRMSE -> {RMSE2} m')

    if mean_f1 > best2dmetric:
      best2dmetric = mean_f1
      torch.save(net.state_dict(), out_file+'2dbestnet.pth')
      print('Best 2D model saved!')

    if RMSE2 < best3dmetric:
      best3dmetric = RMSE2
      torch.save(net.state_dict(), out_file+'3dbestnet.pth')
      print('Best 3D model saved!')

  stats = dict(epoch = epoch, Loss2D = epoch_2d_loss, Loss3D = epoch_3d_loss, LossCon = epoch_con_loss, Loss = epoch_loss, RMSE = RMSE1, cRMSE = RMSE2, F1Score = mean_f1*100)


end = time.time()
print('Training completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
print(f'Best metrics: F1 score -> {best2dmetric*100} %,\t cRMSE -> {best3dmetric}')
