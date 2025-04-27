import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np
import random
import math

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg (verts, labels, path, device):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]
    # print("shape of sample labels: ", labels.shape)
    # print("shape of verts: ", verts.shape)
    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,verts.shape[0],3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15, loop=0)


def Q1_vis(test_label, pred_label, test_data, args):
    ind_0 = []
    ind_incorrect_0 = []
    ind_1 = []
    ind_incorrect_1 = []
    ind_2 = []
    ind_incorrect_2 = []
    for i in range(test_label.shape[0]):
        if test_label[i].item() == 0:
            if pred_label[i].item() == 0:
                ind_0.append(i)
            else:
                ind_incorrect_0.append(i)
        elif test_label[i].item() == 1:
            if pred_label[i].item() == 1:
                ind_1.append(i)
            else:
                ind_incorrect_1.append(i)
        else:
            if pred_label[i].item() == 2:
                ind_2.append(i)
            else:
                ind_incorrect_2.append(i)
    print("Shapes of : ")
    print("ind_0: ", len(ind_0))
    print("ind_incorrect_0: ", len(ind_incorrect_0))
    print("ind_1: ", len(ind_1))
    print("ind_incorrect_1: ", len(ind_incorrect_1))
    print("ind_2: ", len(ind_2))
    print("ind_incorrect_2: ", len(ind_incorrect_2))
    selected = random.sample(ind_0, min(len(ind_0), 3))
    for i in selected:
        print("for index {} - Ground truth: {}, Predicted: {}".format(i, test_label[i], pred_label[i]))
        temp_labels = torch.ones(test_data[i].shape[0])
        viz_seg(test_data[i],temp_labels, "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
    print("Chairs correct done")
    selected = random.sample(ind_incorrect_0, min(len(ind_incorrect_0), 3))
    for i in selected:
        print("for index {} - Ground truth: {}, Predicted: {}".format(i, test_label[i], pred_label[i]))
        temp_labels = torch.ones(test_data[i].shape[0])
        viz_seg(test_data[i],temp_labels, "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
    print("chairs incorrect done")
    selected = random.sample(ind_1, min(len(ind_1), 3))
    for i in selected:
        print("for index {} - Ground truth: {}, Predicted: {}".format(i, test_label[i], pred_label[i]))
        temp_labels = torch.ones(test_data[i].shape[0])
        viz_seg(test_data[i],temp_labels, "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
    print("Lamps correct done")
    selected = random.sample(ind_incorrect_1, min(len(ind_incorrect_1), 3))
    for i in selected:
        print("for index {} - Ground truth: {}, Predicted: {}".format(i, test_label[i], pred_label[i]))
        temp_labels = torch.ones(test_data[i].shape[0])
        viz_seg(test_data[i],temp_labels, "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
    print("Lamps incorrrct done")
    selected = random.sample(ind_2, min(len(ind_2), 3))
    for i in selected:
        print("for index {} - Ground truth: {}, Predicted: {}".format(i, test_label[i], pred_label[i]))
        temp_labels = torch.ones(test_data[i].shape[0])
        viz_seg(test_data[i],temp_labels, "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
    print("Vase correct done")
    selected = random.sample(ind_incorrect_2, min(len(ind_incorrect_2), 3))
    for i in selected:
        print("for index {} - Ground truth: {}, Predicted: {}".format(i, test_label[i], pred_label[i]))
        temp_labels = torch.ones(test_data[i].shape[0])
        viz_seg(test_data[i],temp_labels, "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
    print("Vase inorrect done")

def rotate_x(points, theta_deg):
    B, N, _ = points.shape
    theta = torch.tensor(math.radians(theta_deg), device=points.device)
    
    # Create rotation matrix of shape [3, 3]
    rot_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(theta), -torch.sin(theta)],
        [0, torch.sin(theta),  torch.cos(theta)]
    ], dtype=torch.float32, device=points.device)

    # Expand to [B, 3, 3] to apply per batch
    rot_x = rot_x.unsqueeze(0).expand(B, -1, -1)  # shape: [B, 3, 3]

    # Apply rotation: [B, N, 3] x [B, 3, 3] -> [B, N, 3]
    rotated = torch.bmm(points, rot_x.transpose(1, 2))
    return rotated