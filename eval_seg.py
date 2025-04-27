import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg, rotate_x


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/seg')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--degree', type=int, default=0, help="Degrees to rotate the point cloud about x axis")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model()
    model.to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # Rotate point cloud
    test_data = rotate_x(test_data, args.degree)

    # ------ TO DO: Make Prediction ------
    batch_size =  25
    num_samples = test_data.size(0)

    all_preds = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i+batch_size, num_samples)
        batch = test_data[i:end_idx]

        with torch.no_grad():
            batch_logits = model(batch.to(args.device))
            # print("Shape of batch_logits:", batch_logits.shape)
            batch_preds = batch_logits.max(2)[1]
    
        all_preds.append(batch_preds.cpu())
    
        torch.cuda.empty_cache()

    # Combine all batch predictions
    pred_label = torch.cat(all_preds, dim=0)
    # print("Shape of predictions: ", pred_label[0].shape)
    # print("Shape of test: ", test_label[0].shape)

    index = [0, 20, 117, 170, 26, 41, 90]
    for i in index:
        acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / (test_label[0].reshape((-1,1)).size()[0])
        print("Accuracy for {}: {}".format(i, acc))
        viz_seg(test_data[i], test_label[i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
        viz_seg(test_data[i], pred_label[i], "{}/pred_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)

    # For visualising low accuracy instances

    # failure_cases = []
    # for i in range(pred_label.shape[0]):
    #     acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / (test_label[0].reshape((-1,1)).size()[0])
    #     if acc<0.7:
    #         failure_cases.append(i)
    #         viz_seg(test_data[i], test_label[i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, str(i)), args.device)
    #         viz_seg(test_data[i], pred_label[i], "{}/pred_{}_{}.gif".format(args.output_dir, args.exp_name, str(i)), args.device)
    #     if len(failure_cases)>3:
    #         break
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
