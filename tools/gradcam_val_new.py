"""by wyf"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lib.datasets.kitti_utils import get_affine_transform
from lib.models.MonoASRH import MonoASRH
from lib.datasets.kitti_utils import Calibration


class CustomModel(nn.Module):
    def __init__(self, original_model, coord_ranges, calibs):
        super(CustomModel, self).__init__()
        self.original_model = original_model
        self.coord_ranges = coord_ranges
        self.calibs = calibs

    def forward(self, input_tensor):
        image_tensor = input_tensor
        ret = self.original_model(image_tensor, self.coord_ranges, self.calibs, mode='test')
        heatmap = ret['heatmap']
        return heatmap


class SemanticSegmentationTarget:
    def __init__(self, mask):
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[:, :, :] * self.mask).sum()
    

def get_image(image_dir, idx):
    img_file = os.path.join(image_dir, '%06d.png' % idx)
    assert os.path.exists(img_file)
    return Image.open(img_file)    # (H, W, 3) RGB mode

def get_calib(calib_dir, idx):
    calib_file = os.path.join(calib_dir, '%06d.txt' % idx)
    assert os.path.exists(calib_file)
    return Calibration(calib_file)

def load_checkpoint(model, optimizer, filename, map_location):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=map_location)
        epoch = checkpoint.get('epoch', -1)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(map_location)
        # epoch = 5
    else:
        raise FileNotFoundError
    return epoch


def main(index):
    if not os.path.exists("/media/data3/wangyf/MonoASRH/out_cam"):
        os.makedirs("/media/data3/wangyf/MonoASRH/out_cam")
    save_root = "/media/data3/wangyf/MonoASRH/out_cam"
    split_dir = os.path.join('/', '/media/data3/wangyf/KITTI', 'ImageSets', 'trainval' + '.txt')
    idx_list = [x.strip() for x in open(split_dir).readlines()]
    data_dir = os.path.join('/', '/media/data3/wangyf/KITTI', 'training')
    image_dir = os.path.join(data_dir, 'image_2')
    calib_dir = os.path.join(data_dir, 'calib')
    # Parameters:
    resolution = np.array([1280, 384])  # W * H
    # statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    ##l,w,h
    cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                              [1.52563191462 ,1.62856739989, 3.88311640418],
                              [1.73698127    ,0.59706367   , 1.76282397   ]])      
    data_transform = transforms.Compose([transforms.ToTensor()])

    idx_list = [76]

    for idx in idx_list:
        idx = int(idx)
        print(idx)
        ori_img = get_image(image_dir, idx)
        img_size = np.array(ori_img.size)
        dst_W, dst_H = img_size

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, resolution, inv=1)
        ori_img = ori_img.transform(tuple(resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)
        calib = get_calib(calib_dir, idx).P2
        coord_range, calib = torch.unsqueeze(torch.from_numpy(coord_range), dim=0).cuda(), torch.unsqueeze(torch.torch.from_numpy(calib), dim=0).cuda()
        # image encoding
        img = np.array(ori_img).astype(np.float32) / 255.0
        img = (img - mean) / std
        #img = img.transpose(2, 0, 1)  # C * H * W
        img_tensor = data_transform(img).cuda()
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        if idx == 76:
            model = MonoASRH(backbone='dla34', neck='DLAUp', mean_size=cls_mean_size).cuda()
            load_checkpoint(model = model,
                            optimizer = None,
                            filename = "/media/data3/wangyf/MonoASRH/MonoASRH_Scale2/logs/checkpoints/checkpoint_epoch_385.pth",
                            map_location = input_tensor.device)
            model = CustomModel(model, coord_range, calib).cuda()
            target_layers = [model.original_model.heatmap[0]]
            #target_layers = [model.original_model.feat_up.ida_2.node_3]
            #target_layers = [model.original_model.feat_up.output_proj[-1]]
            #breakpoint()
        
        model.eval()
        output = model(input_tensor)
        normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
        mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        mask_value, _ = normalized_masks[0, :, :, :].max(axis=0)
        final_mask = mask
        final_mask[mask_value < 0.5] = 3
        mask = final_mask
        mask_float = np.float32(mask)

        # Grad CAM
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [SemanticSegmentationTarget(mask_float)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        # 转换为 numpy 格式并放缩到0-1之间
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = np.clip(grayscale_cam, 0, 1)
        # 将 grad-cam 的输出叠加到原始图像上
        ori_img = np.array(ori_img)
        visualization = show_cam_on_image(ori_img.astype(dtype=np.float32)/255, grayscale_cam)

        save_path = os.path.join(save_root, '%06d.png' % idx)
        cv2.imwrite(save_path, visualization)
    #plt.imshow(visualization)
    #plt.show()


    


    pass


if __name__ == "__main__":
    main(0)