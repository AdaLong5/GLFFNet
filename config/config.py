import os
import os.path as osp
import models

# configuration file
description  = 'glffnet'
version_string  = '0.1'

# device can be "cuda" or "gpu"
device = 'cuda'
num_workers = 4
available_gpus = '0'
# available_gpus = '0,2,3'
print_freq = 15

work_root = os.getcwd()
result_root = osp.join(work_root, 'result')
#result_root = osp.join(work_root, 'result_loss')
# result_sub_folder = osp.join(result_root, f'{description}_{version_string}_torch')
ckpt_folder = osp.join(result_root, 'ckpt')

base_model_name = models.ALEXNET
# base_model_name = models.VGG13BN
# base_model_name = models.VGG11BN
# base_model_name = models.RESNET50
# base_model_name = models.INCEPTION_V3


class pc_net:
    #data_root = osp.join(work_root, 'data', 'pc')
    data_root = '/home/jiao/workspace/dataset/modelnet40/pc'

    n_neighbor = 20
    num_classes = 40
    pre_trained_model = None
    # ckpt_file = osp.join(ckpt_folder, 'PCNet-ckpt.pth')
    ckpt_file = osp.join(ckpt_folder, 'PCNet-save-ckpt.pth')
    ckpt_load_file = osp.join(ckpt_folder, 'pc-91.086-0.85701.pth')

    class train:
        batch_sz = 24*4
        resume = False
        resume_epoch = None

        lr = 0.001
        momentum = 0.9
        weight_decay = 0
        max_epoch = 250
        data_aug = True

    class validation:
        batch_sz = 32

    class test:
        batch_sz = 32


class view_net:
    num_classes = 40

    # multi-view cnn
    #data_root = osp.join(work_root, 'data', '12_ModelNet40')
    data_root= '/home/jiao/workspace/dataset/modelnet40/12_ModelNet40'
   

    pre_trained_model = None
    if base_model_name == (models.ALEXNET or models.RESNET50):
        ckpt_file = osp.join(ckpt_folder, f'MVCNN-{base_model_name}-save-ckpt.pth')
        ckpt_load_file = osp.join(ckpt_folder, f'MVCNN-{base_model_name}-save-ckpt.pth')
       
        t = osp.join(ckpt_folder, f'MV.pth')
    else:
        ckpt_file = osp.join(ckpt_folder, f'{base_model_name}-12VIEWS-MAX_POOLING-save-ckpt.pth')
        ckpt_load_file = osp.join(ckpt_folder, f'{base_model_name}-12VIEWS-MAX_POOLING-ckpt.pth')

    class train:
        if base_model_name == models.ALEXNET:
            batch_sz = 128 # AlexNet 2 gpus
        elif base_model_name == models.INCEPTION_V3:
            batch_sz = 2
        else:
            batch_sz = 32
        resume = False
        resume_epoch = None

        lr = 0.001
        momentum = 0.9
        weight_decay = 1e-4
        max_epoch = 200
        data_aug = True

    class validation:
        batch_sz = 256

    class test:
        batch_sz = 32

class glff_net:
    num_classes = 40#10#40
    


    # pointcloud 
 
    pc_root = '/home/jiao/workspace/dataset/modelnet40/pc'
    n_neighbor = 20
    #tsne可视化
    pc_root_tsne = osp.join(work_root, 'data', 'pc_tsne3')

    # multi-view cnn 
    
    view_root = '/home/jiao/workspace/dataset/modelnet40/12_ModelNet40'
    
    #tsne可视化
    view_root_tsne = osp.join(work_root, 'data', '12_ModelNet40_tsne')

    pre_trained_model = False
    ckpt_file = osp.join(ckpt_folder, f'0.5dcl.pth.pth')
    ckpt_load_file = osp.join(ckpt_folder, f'0.5dcl.pth')
    ccl_file = osp.join(ckpt_folder, f'0.5dcl.pth')
    #ccl_file = osp.join(ckpt_folder, f'0.4dcl-93.193-0.92529-256.pth')
    


    class train:
        # optim = 'Adam'
        optim = 'SGD'
        # batch_sz = 18*2
        batch_sz = 20
        batch_sz_res = 5*1
        resume = False
        resume_epoch = None

        iter_train = True
        # iter_train = False

        fc_lr = 0.01
        all_lr = 0.0009
        momentum = 0.9
        # weight_decay = 5e-4
        weight_decay = 1e-5
        max_epoch = 10#100
        data_aug = True

    class validation:
        #batch_sz = 2
        batch_sz = 40

    class test:
        # batch_sz = 2
        batch_sz = 32