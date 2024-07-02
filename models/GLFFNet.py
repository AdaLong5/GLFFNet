
from models import *
import config
import numpy as np
import torch.nn.functional as F

class GLFFNet(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(GLFFNet, self).__init__()

        self.fea_dim = 1024
        self.fea_dim1 = 2048
        self.num_bottleneck = 1024  # 512
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.glff_net.n_neighbor

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)
        self.conv2d5 = conv_2d(320, 1024, 1)

        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc_local = nn.Sequential(
            fc_layer(2048, 1024, True),
        )
        self.fusion_fc = nn.Sequential(
            fc_layer(3072, 1024, True),
        )

        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                fc_layer((scale + 1) * self.fea_dim, self.num_bottleneck, True),
            )

            self.fusion_fc_scales += [fc_fusion]

        self.sig = nn.Sigmoid()
        self.fc = nn.Sequential(
            fc_layer(1024, 512, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp2 = nn.Sequential(
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        if init_weights:
            self.init_mvcnn()
            self.init_dgcnn()
        encoder_layer = nn.TransformerEncoderLayer(512, nhead=2)
        encoder_layer1 = nn.TransformerEncoderLayer(1024, nhead=2)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, num_layers=1)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer1, num_layers=1)

    def init_mvcnn(self):
        print(f'init parameter from mvcnn {config.base_model_name}')
        mvcnn_state_dict = torch.load(config.view_net.ckpt_load_file)['model']
        glffnet_state_dict = self.state_dict()

        mvcnn_state_dict = {k.replace('features', 'mvcnn', 1): v for k, v in mvcnn_state_dict.items()}
        mvcnn_state_dict = {k: v for k, v in mvcnn_state_dict.items() if k in glffnet_state_dict.keys()}
        glffnet_state_dict.update(mvcnn_state_dict)
        self.load_state_dict(glffnet_state_dict)
        print(f'load ckpt from {config.view_net.ckpt_load_file}')

    def init_dgcnn(self):
        print(f'init parameter from dgcnn')
        dgcnn_state_dict = torch.load(config.pc_net.ckpt_file)['model']
        glffnet_state_dict = self.state_dict()

        dgcnn_state_dict = {k: v for k, v in dgcnn_state_dict.items() if k in glffnet_state_dict.keys()}
        glffnet_state_dict.update(dgcnn_state_dict)
        self.load_state_dict(glffnet_state_dict)
        print(f'load ckpt from {config.pc_net.ckpt_file}')

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

        x_edge = get_edge_feature(pc, self.n_neighbor)
        x_trans = self.trans_net(x_edge)
        x = pc.squeeze(-1).transpose(2, 1)
        x = torch.bmm(x, x_trans)
        x = x.transpose(2, 1)

        x1 = get_edge_feature(x, self.n_neighbor)
        x1 = self.conv2d1(x1)
        x1, _ = torch.max(x1, dim=-1, keepdim=True)

        x2 = get_edge_feature(x1, self.n_neighbor)
        x2 = self.conv2d2(x2)
        x2, _ = torch.max(x2, dim=-1, keepdim=True)

        x3 = get_edge_feature(x2, self.n_neighbor)
        x3 = self.conv2d3(x3)
        x3, _ = torch.max(x3, dim=-1, keepdim=True)

        x4 = get_edge_feature(x3, self.n_neighbor)
        x4 = self.conv2d4(x4)
        x4, _ = torch.max(x4, dim=-1, keepdim=True)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv2d5(x5)
        x5, _ = torch.max(x5, dim=-2, keepdim=True)

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)

        pc = x5.squeeze()
       
        pc_expand1 = pc.unsqueeze(1).expand(-1, view_num, -1)
       
        pc_expand = pc_expand1.contiguous().view(batch_size * view_num, -1)
    

        # # Get Relation Scores--
      
        dist = F.cosine_similarity(mv_view, pc_expand, dim=1)
        dist = dist.view(batch_size, view_num, -1)
        fusion_mask = self.sig(dist)
        fusion_mask = torch.softmax(fusion_mask, 1, torch.float32)
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        # Enhance View Feature
        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        
        scale_out = []
        for i in range(len(self.n_scale)):
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size,
                                                                                                   self.n_scale[
                                                                                                       i] * self.fea_dim)
            pc_mv_scale = torch.cat((pc, mv_scale_fea), dim=1)
            
            pc_mv_scale = self.fusion_fc_scales[i](pc_mv_scale)
           
            scale_out.append(pc_mv_scale.unsqueeze(2))
        #scale_out = torch.cat(scale_out, dim=2).mean(2)
        scale_out = torch.cat(scale_out, dim=2).max(2)[0]
 

        mv_global = torch.max(mv_view_expand.view(batch_size, view_num, self.num_bottleneck), dim=1)[0]
        
        # pc_global = torch.max(pc_expand1.view(batch_size, view_num, self.num_bottleneck), dim=1)[0]
        pc_global = pc

        
        matrix = torch.stack((pc_global, mv_global, scale_out), dim=0)

        feature_fuse = self.transformer_encoder1(matrix)
        # # print(feature_fuse)
        final_out = torch.cat((feature_fuse[0], feature_fuse[1], feature_fuse[2]), dim=1)

        final_out = self.fusion_fc(final_out) #+ torch.max(matrix.view(3, batch_size, self.num_bottleneck), dim=0)[0]  # 3072->1024

        # Final FC Layers
        net_fea = self.fusion_mlp2(scale_out)
        net = self.fusion_mlp3(net_fea)
        

        if get_fea:
            return net, net_fea
            # return net, final_out, pc_expand, mv_view

        else:
            return net

