import torch
import torchvision
from utils.networks import *
import torchvision.datasets as datasets
import os
import argparse
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.utils.linear_assignment_ import linear_assignment
import utils
import cv2

dtype = torch.torch.FloatTensor
# marginalized entropy
def m_entropy(y):
    #y1 = Variable(torch.randn(y.size(1)).type(dtype), requires_grad=True)
    #y2 = Variable(torch.randn(1).type(dtype), requires_grad=True)
    y1 = y.mean(0)
    y2 = -torch.sum(y1*torch.log(y1+ 1e-7))
    return y2

# conditional entropy
def c_entropy(y,arg):
    #y1 = Variable(torch.randn(y.size()).type(dtype), requires_grad=True)
    #y2 = Variable(torch.randn(1).type(dtype), requires_grad=True)
    y1 = -y*torch.log(y + 1e-7)
    y2 = 1.0/arg.batch_size*y1.sum()
    return y2

def adujest_lr(optimizer,ini_lr,epoch,epoch_num):
    for p in optimizer.param_groups:
        if epoch < 5:
            lr = 0.1*ini_lr
            p['lr'] = lr
        elif epoch <0.8*epoch_num:
            p['lr'] = ini_lr
        else:
            lr = 0.1*ini_lr
            p['lr'] = lr


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


def evaluator(val_loader,d_net,gpu_use):
    acc_all = 0
    for i, (r_img, label) in enumerate(val_loader,1):
        if gpu_use:
            r_img = Variable(r_img.cuda())
        label = label.cpu().data.numpy()
        pre_y = d_net(r_img)
        pre_y = torch.argmax(pre_y,dim=-1).cpu().data.numpy()
        acc,_ = cluster_acc(pre_y, label)
        acc_all += acc
    return acc_all/i

def main(arg):
    #########################
    # prepare the dataset
    #########################
    """
    one = torch.FloatTensor([1])
    mone = -1.*one
    """
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    train_path = os.path.join('MNIST','mnist_train')
    val_path = os.path.join('MNIST','mnist_validation')
    train_set = datasets.MNIST(train_path,download=True,train=True,transform=transform)
    val_set = datasets.MNIST(val_path,download=True,train=False,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=arg.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=arg.batch_size,shuffle=True)

    ########################
    # create the network
    ########################
    gpu_use = torch.cuda.is_available()
    g_net = generator_v1(arg.noise_dim)
    d_net = discriminator_v1()

    if arg.resume:
        if arg.g_path=='' or arg.d_path=='':
            print('no checkpoint path')
            sys.exit(0)
        g_net.load_state_dict(torch.load(arg.g_path))
        d_net.load_state_dict(torch.load(arg.d_path))

    if gpu_use:
        g_net = g_net.cuda()
        d_net = d_net.cuda()
        # one = one.cuda()
        # mone = mone.cuda()

    # get the to be trained parameters
    g_params = g_net.parameters()
    d_params = d_net.parameters()

    ########################
    # train the network
    ########################

    # define the optimizer
    if arg.optimizer == 'Adam':
        g_optimizer = optim.Adam(params=g_params,lr=arg.lr)
        d_optimizer = optim.Adam(params=d_params,lr=arg.lr)
    elif arg.optimizer == 'SGD':
        g_optimizer = optim.SGD(params=g_params,lr=arg.lr)
        d_optimizer = optim.SGD(params=d_params,lr=arg.lr)


    # training
    it_index = 0
    for epoch in range(arg.epoch_num):
        for it, (r_img,label) in enumerate(train_loader):
            # update the discriminator
            adujest_lr(g_optimizer,arg.lr,epoch,arg.epoch_num)
            adujest_lr(d_optimizer, arg.lr, epoch, arg.epoch_num)
            for p in d_net.parameters():
                p.required_grad = True
            for p in g_net.parameters():
                p.required_grad = False

            d_net.zero_grad()
            if gpu_use:
                r_img = Variable(r_img.cuda())
            pre_r = d_net(r_img)
            c_entropy_r = c_entropy(pre_r,arg)
            #c_entropy_r.backward(one,retain_graph=True)

            m_entropy_r = m_entropy(pre_r)
            #m_entropy_r.backward(mone)

            z = torch.randn((arg.batch_size,arg.noise_dim))
            if gpu_use:
                z = Variable(z.cuda())

            f_img = g_net(z)
            pre_f = d_net(f_img)
            c_entropy_f = c_entropy(pre_f,arg)
            #c_entropy_f.backward(mone)

            l_d = c_entropy_r - m_entropy_r - c_entropy_f
            l_d.backward(retain_graph=True)
            d_optimizer.step()


            #update the generator
            for p in g_net.parameters():
                p.required_grad = True
            for p in d_net.parameters():
                p.required_grad = False
            g_net.zero_grad()
            """
            z = torch.randn((arg.batch_size, arg.noise_dim))
            if gpu_use:
                z = Variable(z.cuda())
            
            f_img = g_net(z)
            pre_f = d_net(f_img)
            
            c_entropy_f = c_entropy(pre_f,arg)
            #c_entropy_f.backward(one,retain_graph=True)
            """
            m_entropy_f = m_entropy(pre_f)
            #m_entropy_f.backward(mone)

            l_g = c_entropy_f - m_entropy_f

            l_g.backward(retain_graph=True)
            g_optimizer.step()

            l_d = l_d.cpu().data.numpy()
            l_g = l_g.cpu().data.numpy()

            if it % 500 == 0:
                #print(epoch,'\t',l_d,'\t',l_g)
                print('it %s [epoch %s] \t %s %.4f \t %s %.4f' %
                      (str(it_index).zfill(3),str(epoch).zfill(2),'d_loss',l_d,'g_loss',l_g))
            it_index += 1
        ################################################
        # evaluation and saving results
        ################################################

        # calculate the accuracy
        acc = evaluator(val_loader,d_net,gpu_use)
        print('the accuracy of epoch %d is \t %.4f' % (epoch,acc))

        # save state per epoch
        if not os.path.exists(os.path.join(arg.save_log,'checkpoint')):
            os.makedirs(os.path.join(arg.save_log,'checkpoint'))
        torch.save(g_net.state_dict(), os.path.join(arg.save_log, 'checkpoint', 'g_net_last.pth'))
        torch.save(d_net.state_dict(), os.path.join(arg.save_log, 'checkpoint', 'd_net_last.pth'))

        if epoch % 10 == 0:
            torch.save(g_net.state_dict(),os.path.join(arg.save_log,'checkpoint','g_net'+str(epoch)+'.pth'))
            torch.save(d_net.state_dict(),os.path.join(arg.save_log,'checkpoint','d_net'+str(epoch)+'.pth'))

        # save images
        z_save = torch.rand((arg.batch_size, arg.noise_dim))
        if torch.cuda.is_available():
            z_save = z_save.cuda()
        f_img_save = g_net(Variable(z_save)).data.cpu().numpy()

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        save_path = os.path.join(arg.save_log, 'imges',str(epoch))
        if not os.path.exists(os.path.join( save_path)):
            os.makedirs(os.path.join( save_path))
        for i, sample in enumerate(f_img_save):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_aspect('equal')
            sample_new =sample.transpose((1, 2, 0)) * 0.5 + 0.5
            plt.imshow(np.concatenate((sample_new,sample_new,sample_new), axis=-1), cmap='gray')
        img_name = str(epoch).zfill(3) + '.png'
        plt.savefig(os.path.join(save_path, img_name))
        plt.close(fig)

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--epoch_num', type=int, default=100, help='the number of epochs')
    arg.add_argument('--batch_size', type=int, default=64, help='the batch size for training')
    arg.add_argument('--noise_dim', type=int, default=128, help='the dimension of the noise vector')
    arg.add_argument('--lr', type=float, default=2e-4, help='the learning rate for updating model')
    arg.add_argument('--itr_dis', type=int, default=1,
                     help='updata d-params for itr_dis times for every update of generator')
    arg.add_argument('--optimizer', type=str, default='Adam', help='the optimization method for updating parameters')
    arg.add_argument('--save_log', type=str, default='log', help='the path for saving results, parameters and model')
    arg.add_argument('--resume',action='store_true',help='determine if it continues to train')
    arg.add_argument('--g_path',type=str,default='',help='the path for generator checkpoint')
    arg.add_argument('--d_path', type=str, default='', help='the path for discriminator checkpoint')

    main(arg.parse_args())
