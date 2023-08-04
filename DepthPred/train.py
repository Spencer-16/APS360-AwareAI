import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math

def custom_loss(output, gt_depth):
    # First term: L1 loss of original images
    loss1 = nn.L1Loss()
    l_depth = loss1(output, gt_depth)
    # Second term: L2 loss(MSE)
    loss2 = nn.MSELoss()
    l_ssim = loss2(output, gt_depth)
    loss = l_ssim + l_depth
    return loss

def RMSE(output, gt_depth):
    assert output.shape == gt_depth.shape, \
        "Output and gt_depth sizes don't match!"
    diff = output - gt_depth
    diff2 = torch.pow(diff, 2)
    mse = torch.mean(diff2)
    rmse = torch.sqrt(mse)
    return rmse

def RMSE_log(output, gt_depth):
    assert output.shape == gt_depth.shape, \
        "Output and gt_depth sizes don't match!"
    output_log = torch.log(torch.clamp(output, min=1e-4, max=80.0))
    gt_depth_log = torch.log(torch.clamp(gt_depth, min=1e-4, max=80.0))
    diff_log = output_log - gt_depth_log
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))
    # diff2 = torch.pow(diff_log, 2)
    # mse_log = torch.div(torch.sum(diff2), output.numel())
    # rmse_log = torch.sqrt(mse_log)
    return rmse_log

def ABS_rel(output, gt_depth):
    assert output.shape == gt_depth.shape, \
        "Output and gt_depth sizes don't match!"
    gt_depth = torch.clamp(gt_depth, min=1e-4, max=80.0)
    diff_abs = torch.abs(output - gt_depth)
    abs_rel = torch.mean(diff_abs / gt_depth)
    # diff_rel = torch.div(diff_abs, gt_depth)
    # abs_rel = torch.div(torch.sum(diff_rel), output.numel())
    return abs_rel

def SQ_rel(output, gt_depth):
    assert output.shape == gt_depth.shape, \
        "Output and gt_depth sizes don't match!"
    gt_depth = torch.clamp(gt_depth, min=1e-4, max=80.0)
    diff = output - gt_depth
    sq_rel = torch.mean(torch.pow(diff, 2) / gt_depth)
    # diff2 = torch.pow(diff, 2)
    # diff_rel = torch.div(diff2, gt_depth)
    # sq_rel = torch.div(torch.sum(diff_rel), output.numel())
    return sq_rel

def eval_threshold_acc(output, gt_depth, threeshold_val):
    assert output.shape == gt_depth.shape, \
        "Output and gt_depth sizes don't match!"
    thresh = torch.max((gt_depth / output), (output / gt_depth))

    d1 = torch.sum(thresh < threeshold_val).float() / len(thresh)
    d2 = torch.sum(thresh < threeshold_val ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < threeshold_val ** 3).float() / len(thresh)
    return d1, d2, d3

def get_file_name(exp_name, model_name, epoch, lr, bs):
    return exp_name + '_' + model_name + '_ep' + str(epoch) + '_lr' + str(lr) + '_bs' + str(bs)

def plot_losses(path):
    import matplotlib.pyplot as plt
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    n = len(train_loss) # number of epochs
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def plot_metrics(path, metric):
    import matplotlib.pyplot as plt
    assert metric in ['rmse', 'rmse_log', 'abs_rel', 'sq_rel'], "No such metric"
    train_metric = np.loadtxt("{}_train_".format(path)+metric+".csv")
    val_metric = np.loadtxt("{}_val_".format(path)+metric+".csv")
    n = len(train_metric) # number of epochs
    plt.title("Train vs Validation: "+metric)
    plt.plot(range(1,n+1), train_metric, label="Train")
    plt.plot(range(1,n+1), val_metric, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend(loc='best')
    plt.show()

def train_coarse(model, train_loader, val_loader, epoch, lr, bs, exp_name):
    root_pth = "/content/gdrive/MyDrive/APS360-AwareAI/DepthPred/"
    model.train()
    model.cuda()
    device = torch.device('cuda')
    best_metric = 99999999
    train_coarse_loss = np.zeros(epoch)
    eval_coarse_loss = np.zeros(epoch)
    train_rmse, train_rmse_log, train_abs_rel, train_sq_rel = \
        np.zeros(epoch), np.zeros(epoch), np.zeros(epoch), np.zeros(epoch)
    eval_rmse, eval_rmse_log, eval_abs_rel, eval_sq_rel = \
        np.zeros(epoch), np.zeros(epoch), np.zeros(epoch), np.zeros(epoch)
    coarse_optimizer = optim.Adam(model.parameters(), lr = lr)
    print("Start training: ")
    start_time = time.time()
    for e in range(epoch):
        for batch_idx, data in enumerate(train_loader):
            rgb = data[0].to(device)
            depth = data[1].to(device)
            coarse_optimizer.zero_grad()
            output = model(rgb)
            depth = depth.view(-1, 1, 60, 80)

            # compute loss and metrics
            loss = custom_loss(output, depth)
            rmse = RMSE(output, depth)
            rmse_log = RMSE_log(output, depth)
            abs_rel = ABS_rel(output, depth)
            sq_rel = SQ_rel(output, depth)

            # update parameters and log metrics
            loss.backward()
            coarse_optimizer.step()
            train_coarse_loss[e] += loss.item()
            train_rmse[e] += rmse.item()
            train_rmse_log[e] += rmse_log.item()
            train_abs_rel[e] += abs_rel.item()
            train_sq_rel[e] += sq_rel.item()

        eval_loss, rmse, rmse_log, abs_rel, sq_rel = eval_coarse(model, val_loader)
        eval_coarse_loss[e] = eval_loss
        eval_rmse[e] = rmse
        eval_rmse_log[e] = rmse_log
        eval_abs_rel[e] = abs_rel
        eval_sq_rel[e] = sq_rel

        # Compute the average of metrics
        train_coarse_loss[e] /= (batch_idx+1)
        train_rmse[e] /= (batch_idx+1)
        train_rmse_log[e] /= (batch_idx+1)
        train_abs_rel[e] /= (batch_idx+1)
        train_sq_rel[e] /= (batch_idx+1)

        end_time = time.time()
        elapsed_time = end_time - start_time
        if e==0 or (e+1)%5 == 0:
            print(f"Epoch {e+1}: Average training loss is {train_coarse_loss[e]} "
                  f"Eval loss is {eval_coarse_loss[e]} "
                  f"Time elapsed: {elapsed_time:.2f} seconds")
        if (rmse+rmse_log) < best_metric:
            best_metric = rmse+rmse_log
            file_name = get_file_name(exp_name, model.name, epoch, lr, bs)
            torch.save(model.state_dict(), root_pth+"pretrained_models/"+file_name+".pt")
    # store all metrics and losses for visualization
    file_name = get_file_name(exp_name, model.name, epoch, lr, bs)
    np.savetxt(root_pth+"logs/"+"{}_train_loss.csv".format(file_name), train_coarse_loss)
    np.savetxt(root_pth+"logs/"+"{}_val_loss.csv".format(file_name), eval_coarse_loss)
    np.savetxt(root_pth+"logs/"+"{}_train_rmse.csv".format(file_name), train_rmse)
    np.savetxt(root_pth+"logs/"+"{}_val_rmse.csv".format(file_name), eval_rmse)
    np.savetxt(root_pth+"logs/"+"{}_train_rmse_log.csv".format(file_name), train_rmse_log)
    np.savetxt(root_pth+"logs/"+"{}_val_rmse_log.csv".format(file_name), eval_rmse_log)
    np.savetxt(root_pth+"logs/"+"{}_train_abs_rel.csv".format(file_name), train_abs_rel)
    np.savetxt(root_pth+"logs/"+"{}_val_abs_rel.csv".format(file_name), eval_abs_rel)
    np.savetxt(root_pth+"logs/"+"{}_train_sq_rel.csv".format(file_name), train_sq_rel)
    np.savetxt(root_pth+"logs/"+"{}_val_sq_rel.csv".format(file_name), eval_sq_rel)

def eval_coarse(model, val_loader):
    model.eval()
    model.cuda()
    device = torch.device('cuda')
    eval_coarse_loss, eval_rmse, eval_rmse_log, eval_abs_rel, eval_sq_rel = 0,0,0,0,0
    for batch_idx, data in enumerate(val_loader):
        rgb = data[0].to(device)
        depth = data[1].to(device)
        depth = depth.view(-1, 1, 60, 80)
        output = model(rgb)

        loss = custom_loss(output, depth)
        rmse = RMSE(output, depth)
        rmse_log = RMSE_log(output, depth)
        abs_rel = ABS_rel(output, depth)
        sq_rel = SQ_rel(output, depth)

        eval_coarse_loss += loss.item()
        eval_rmse += rmse.item()
        eval_rmse_log += rmse_log.item()
        eval_abs_rel += abs_rel.item()
        eval_sq_rel += sq_rel.item()

    model.train()

    eval_coarse_loss /= (batch_idx+1)
    eval_rmse /= (batch_idx+1)
    eval_rmse_log /= (batch_idx+1)
    eval_abs_rel /= (batch_idx+1)
    eval_sq_rel /= (batch_idx+1)

    return eval_coarse_loss, eval_rmse, eval_rmse_log, eval_abs_rel, eval_sq_rel


def train_fine(model, coarse_model, train_loader, val_loader, epoch, lr, bs, exp_name):
    root_pth = "/content/gdrive/MyDrive/APS360-AwareAI/DepthPred/"
    model.train()
    model.cuda()
    coarse_model.eval()
    coarse_model.cuda()
    device = torch.device('cuda')

    best_metric = 99999999
    train_fine_loss = np.zeros(epoch)
    eval_fine_loss = np.zeros(epoch)
    train_rmse, train_rmse_log, train_abs_rel, train_sq_rel = \
        np.zeros(epoch), np.zeros(epoch), np.zeros(epoch), np.zeros(epoch)
    eval_rmse, eval_rmse_log, eval_abs_rel, eval_sq_rel = \
        np.zeros(epoch), np.zeros(epoch), np.zeros(epoch), np.zeros(epoch)

    fine_optimizer = optim.Adam(model.parameters(), lr = lr)
    print("Start training: ")
    start_time = time.time()
    for e in range(epoch):
        for batch_idx, data in enumerate(train_loader):
            rgb = data[0].to(device)
            depth = data[1].to(device)
            fine_optimizer.zero_grad()

            coarse_output = coarse_model(rgb)
            fine_output = model(rgb, coarse_output)
            depth = depth.view(-1, 1, 60, 80)

            # compute loss and metrics
            loss = custom_loss(fine_output, depth)
            rmse = RMSE(fine_output, depth)
            rmse_log = RMSE_log(fine_output, depth)
            abs_rel = ABS_rel(fine_output, depth)
            sq_rel = SQ_rel(fine_output, depth)

            # update parameters and log metrics
            loss.backward()
            fine_optimizer.step()
            train_fine_loss[e] += loss.item()
            train_rmse[e] += rmse.item()
            train_rmse_log[e] += rmse_log.item()
            train_abs_rel[e] += abs_rel.item()
            train_sq_rel[e] += sq_rel.item()

        eval_loss, rmse, rmse_log, abs_rel, sq_rel = eval_fine(model, coarse_model, val_loader)
        eval_fine_loss[e] = eval_loss
        eval_rmse[e] = rmse
        eval_rmse_log[e] = rmse_log
        eval_abs_rel[e] = abs_rel
        eval_sq_rel[e] = sq_rel

        # Compute the average of metrics
        train_fine_loss[e] /= (batch_idx+1)
        train_rmse[e] /= (batch_idx+1)
        train_rmse_log[e] /= (batch_idx+1)
        train_abs_rel[e] /= (batch_idx+1)
        train_sq_rel[e] /= (batch_idx+1)

        end_time = time.time()
        elapsed_time = end_time - start_time

        if e==0 or (e+1)%10 == 0:
            print(f"Epoch {e+1}: Average training loss is {train_fine_loss[e]} "
                  f"Eval loss is {eval_fine_loss[e]} "
                  f"Time elapsed: {elapsed_time:.2f} seconds")
        if (rmse+rmse_log) < best_metric:
            best_metric = rmse+rmse_log
            file_name = get_file_name(exp_name, model.name, epoch, lr, bs)
            torch.save(model.state_dict(), root_pth+"pretrained_models/"+file_name+".pt")

    # store all metrics and losses for visualization
    file_name = get_file_name(exp_name, model.name, epoch, lr, bs)
    np.savetxt(root_pth+"logs/"+"{}_train_loss.csv".format(file_name), train_fine_loss)
    np.savetxt(root_pth+"logs/"+"{}_val_loss.csv".format(file_name), eval_fine_loss)
    np.savetxt(root_pth+"logs/"+"{}_train_rmse.csv".format(file_name), train_rmse)
    np.savetxt(root_pth+"logs/"+"{}_val_rmse.csv".format(file_name), eval_rmse)
    np.savetxt(root_pth+"logs/"+"{}_train_rmse_log.csv".format(file_name), train_rmse_log)
    np.savetxt(root_pth+"logs/"+"{}_val_rmse_log.csv".format(file_name), eval_rmse_log)
    np.savetxt(root_pth+"logs/"+"{}_train_abs_rel.csv".format(file_name), train_abs_rel)
    np.savetxt(root_pth+"logs/"+"{}_val_abs_rel.csv".format(file_name), eval_abs_rel)
    np.savetxt(root_pth+"logs/"+"{}_train_sq_rel.csv".format(file_name), train_sq_rel)
    np.savetxt(root_pth+"logs/"+"{}_val_sq_rel.csv".format(file_name), eval_sq_rel)

def eval_fine(model, coarse_model, val_loader):
    model.eval()
    model.cuda()
    coarse_model.eval()
    coarse_model.cuda()
    device = torch.device('cuda')
    eval_fine_loss, eval_rmse, eval_rmse_log, eval_abs_rel, eval_sq_rel = 0,0,0,0,0
    for batch_idx, data in enumerate(val_loader):
        rgb = data[0].to(device)
        depth = data[1].to(device)
        depth = depth.view(-1, 1, 60, 80)
        coarse_output = coarse_model(rgb)
        output = model(rgb, coarse_output)

        loss = custom_loss(output, depth)
        rmse = RMSE(output, depth)
        rmse_log = RMSE_log(output, depth)
        abs_rel = ABS_rel(output, depth)
        sq_rel = SQ_rel(output, depth)

        eval_fine_loss += loss.item()
        eval_rmse += rmse.item()
        eval_rmse_log += rmse_log.item()
        eval_abs_rel += abs_rel.item()
        eval_sq_rel += sq_rel.item()

    model.train()
    eval_fine_loss /= (batch_idx+1)
    eval_rmse /= (batch_idx+1)
    eval_rmse_log /= (batch_idx+1)
    eval_abs_rel /= (batch_idx+1)
    eval_sq_rel /= (batch_idx+1)

    return eval_fine_loss, eval_rmse, eval_rmse_log, eval_abs_rel, eval_sq_rel