import numpy as np
import argparse

import torch
import random
from models import local_cls_model
from utils import create_dir, viz_seg, Q1_vis, rotate_x
from pytorch3d.transforms import RotateAxisAngle

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/loc_cls')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--degree', type=int, default=0, help="angle to rotate the point cloud")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = local_cls_model()
    model.to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/locality/cls_loc/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))
    # print("Shape of test label: ", test_label.shape)
    # print("Shape of test data: ", test_data.shape)
    # Rotating Point cloud
    # ang_radian = torch.tensor(args.degree * torch.pi / 180.0)
    # R = RotateAxisAngle(axis=torch.tensor([0.0, 1.0, 0.0]), angle=ang_radian)
    # rotated_point_cloud = R.transform_points(test_data)

    # Rotating Point Cloud
    test_data = rotate_x(test_data, args.degree)
    print("Shape of rotated points:", test_data.shape)
    
    # ------ TO DO: Make Prediction ------
    batch_size =  25
    num_samples = test_data.size(0)

    all_preds = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i+batch_size, num_samples)
        batch = test_data[i:end_idx]

        with torch.no_grad():  
            batch_logits = model(batch.to(args.device))
            batch_preds = batch_logits.max(1)[1]
    
        all_preds.append(batch_preds.cpu())
    
        torch.cuda.empty_cache()

    # Combine all batch predictions
    pred_label = torch.cat(all_preds, dim=0)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    print("Visualisation: ")
    # Q1_vis(test_label=test_label, pred_label=pred_label, test_data=test_data, args=args)
    index = [0, 124, 535, 670, 724, 786, 883]
    for i in index:
        print("for index {} - Ground truth: {}, Predicted: {}".format(i, test_label[i], pred_label[i]))
        temp_labels = torch.ones(test_data[i].shape[0])
        viz_seg(test_data[i],temp_labels, "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)