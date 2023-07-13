import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math

def get_file_name(exp_name, model_name, epoch, lr, bs):
    return exp_name + '_' + model_name + '_' + str(bs) + '_' + str(epoch) + '_' + str(lr)

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

def train_coarse(model, train_loader, val_loader, epoch, lr, bs, exp_name):
    root_pth = "/home/spencer/course_repos/APS360-AwareAI/"
    model.train()
    model.cuda()
    device = torch.device('cuda')
    train_coarse_loss = np.zeros(epoch)
    eval_coarse_loss = np.zeros(epoch)
    coarse_optimizer = optim.Adam(model.parameters(), lr = lr)
    print("Start training: ")
    start_time = time.time()
    for e in range(epoch):
        for batch_idx, data in enumerate(train_loader):
            rgb = data[0].to(device)
            depth = data[1].to(device)
            coarse_optimizer.zero_grad()
            output = model(rgb)
            loss = custom_loss(output, depth, rgb.shape[1], rgb.shape[2])
            loss.backward()
            coarse_optimizer.step()
            train_coarse_loss[e] += loss.item()
        eval_loss = eval_course(model, val_loader)

        eval_coarse_loss[e] = eval_loss
        train_coarse_loss[e] /= (batch_idx+1)
        print(f"Epoch {e+1}: Average training loss is {train_coarse_loss[e]} "
              f"Eval loss is {eval_coarse_loss[e]}")

        file_name = get_file_name(exp_name, model.name, e, bs, lr)
        torch.save(model.state_dict(), root_pth+"pretrained_models/"+file_name)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time elapsed: {:.2f} seconds".format(elapsed_time))
    
    file_name = get_file_name(exp_name, model.name, epoch, bs, lr)
    np.savetxt(root_pth+"logs/"+"{}_train_loss.csv".format(file_name), train_coarse_loss)
    np.savetxt(root_pth+"logs/"+"{}_val_loss.csv".format(file_name), eval_coarse_loss)

def eval_course(model, val_loader):
    model.eval()
    model.cuda()
    device = torch.device('cuda')
    eval_course_loss = 0
    for batch_idx, data in enumerate(val_loader):
        rgb = data[0].to(device)
        depth = data[1].to(device)
        output = model(rgb)
        loss = custom_loss(output, depth, rgb.shape[1], rgb.shape[2])
        eval_course_loss += loss.item()
    model.train()
    return eval_course_loss/(batch_idx+1)

def train_fine(model, coarse_model, train_loader, val_loader, epoch, lr, bs, exp_name):
    root_pth = "/home/spencer/course_repos/APS360-AwareAI/"
    model.train()
    coarse_model.eval()
    dtype = torch.cuda.FloatTensor
    train_fine_loss = np.zeros(epoch)
    eval_fine_loss = np.zeros(epoch)
    fine_optimizer = optim.Adam(model.parameters(), lr = lr)
    print("Start training: ")
    start_time = time.time()
    for e in range(epoch):
        for batch_idx, data in enumerate(train_loader):
            rgb = data[0].clone().detach().requires_grad_(True)
            depth = data[2].clone().detach().requires_grad_(True)
            rgb.cuda()
            depth.cuda()
            # rgb = torch.tensor(data[0].cuda(), requires_grad=True)
            # depth = torch.tensor(data[2].cuda(), requires_grad=True)
            fine_optimizer.zero_grad()

            coarse_output = coarse_model(rgb.type(dtype))
            output = model(rgb.type(dtype), coarse_output.type(dtype).cuda()).cuda()

            loss = custom_loss(output, depth, rgb.shape[1], rgb.shape[2])
            loss.backward()
            fine_optimizer.step()
            train_fine_loss[e] += loss.item()
        eval_loss = eval_fine(model, coarse_model, val_loader)

        eval_fine_loss[e] = eval_loss
        train_fine_loss[e] /= (batch_idx+1)
        print(f"Epoch {e+1}: Average training loss is {train_fine_loss[e]} "
              f"Eval loss is {eval_fine_loss[e]}")

        file_name = get_file_name(exp_name, model.name, e, bs, lr)
        torch.save(model.state_dict(), root_pth+"pretrained_models/"+file_name)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time elapsed: {:.2f} seconds".format(elapsed_time))
    
    file_name = get_file_name(exp_name, model.name, epoch, bs, lr)
    np.savetxt(root_pth+"logs/"+"{}_train_loss.csv".format(file_name), train_fine_loss)
    np.savetxt(root_pth+"logs/"+"{}_val_loss.csv".format(file_name), eval_fine_loss)

def eval_fine(model, coarse_model, val_loader):
    model.eval()
    dtype = torch.cuda.FloatTensor
    eval_course_loss = 0
    for batch_idx, data in enumerate(val_loader):
        rgb = torch.tensor(data['image'].cuda())
        depth = torch.tensor(data['depth'].cuda())
        coarse_output = coarse_model(rgb.type(dtype))
        output = model(rgb.type(dtype), coarse_output.type(dtype))
        loss = custom_loss(output, depth, rgb.shape[1], rgb.shape[2])
        eval_course_loss += loss.item()
    model.train()
    return eval_course_loss/(batch_idx+1)

def custom_loss(output, gt_depth, width, height):
    di = gt_depth - output
    n = width * height
    di2 = torch.pow(di, 2)
    first_term = torch.sum(di2, (1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di, (1,2,3)), 2)/ (n**2)
    loss = first_term + second_term
    return loss.mean()