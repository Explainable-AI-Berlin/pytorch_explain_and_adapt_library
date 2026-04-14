"""
HierViT model

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/HierViT`
"""

import torch
from torch import nn
from collections import OrderedDict
from torchvision.models.vision_transformer import vit_b_16, Encoder, MLPBlock, vit_h_14
from torchvision.models import ViT_B_16_Weights
from functools import partial
from typing import Callable

class DecoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class CustomVisionTransformer(nn.Module): # modified from https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L160
    def __init__(self, args):
        super(CustomVisionTransformer, self).__init__()
        self.threeD = args.threeD
        if args.dataset == "derm7pt":
            self.input_size = [3, *args.resize_shape]
        else:
            self.input_size = [1, *args.resize_shape]
        self.num_classes = 1
        if args.dataset == "LIDC":
            self.numcaps = 8
        elif args.dataset == "derm7pt":
            self.numcaps = 7
        elif args.dataset == "FunnyNodules":
            self.numcaps = 6
        self.numProtos = args.num_protos

        # pre trained
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # encoder for each attri
        num_heads = 12
        vitdropout = self.vit.dropout
        vitattentiondropout = self.vit.attention_dropout
        vitmlpdim = self.vit.mlp_dim
        vithiddendim = self.vit.hidden_dim
        if self.numcaps == 8:
            self.encoder_attri0 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri1 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri2 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri3 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri4 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri5 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri6 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri7 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
        if self.numcaps == 7:
            self.encoder_attri0 = Encoder(self.vit.seq_length, 1, num_heads, vithiddendim, vitmlpdim, vitdropout, vitattentiondropout, self.vit.norm_layer)
            self.encoder_attri1 = Encoder(self.vit.seq_length, 1, num_heads, vithiddendim, vitmlpdim, vitdropout, vitattentiondropout, self.vit.norm_layer)
            self.encoder_attri2 = Encoder(self.vit.seq_length, 1, num_heads, vithiddendim, vitmlpdim, vitdropout, vitattentiondropout, self.vit.norm_layer)
            self.encoder_attri3 = Encoder(self.vit.seq_length, 1, num_heads, vithiddendim, vitmlpdim, vitdropout, vitattentiondropout, self.vit.norm_layer)
            self.encoder_attri4 = Encoder(self.vit.seq_length, 1, num_heads, vithiddendim, vitmlpdim, vitdropout, vitattentiondropout, self.vit.norm_layer)
            self.encoder_attri5 = Encoder(self.vit.seq_length, 1, num_heads, vithiddendim, vitmlpdim, vitdropout, vitattentiondropout, self.vit.norm_layer)
            self.encoder_attri6 = Encoder(self.vit.seq_length, 1, num_heads, vithiddendim, vitmlpdim, vitdropout, vitattentiondropout, self.vit.norm_layer)
        if self.numcaps == 6:
            self.encoder_attri0 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri1 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri2 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri3 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri4 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
            self.encoder_attri5 = Encoder(self.vit.seq_length,1,num_heads, vithiddendim,
                                          vitmlpdim,vitdropout,vitattentiondropout,self.vit.norm_layer)
        if self.numcaps == 8:
            self.head_attri0 = nn.Linear(vithiddendim, 1)
            self.head_attri1 = nn.Linear(vithiddendim, 1)
            self.head_attri2 = nn.Linear(vithiddendim, 1)
            self.head_attri3 = nn.Linear(vithiddendim, 1)
            self.head_attri4 = nn.Linear(vithiddendim, 1)
            self.head_attri5 = nn.Linear(vithiddendim, 1)
            self.head_attri6 = nn.Linear(vithiddendim, 1)
            self.head_attri7 = nn.Linear(vithiddendim, 1)
        if self.numcaps == 7:
            self.head_attri0 = nn.Sequential(nn.Linear(vithiddendim, 3), nn.Softmax(dim=-1))
            self.head_attri1 = nn.Sequential(nn.Linear(vithiddendim, 2), nn.Softmax(dim=-1))
            self.head_attri2 = nn.Sequential(nn.Linear(vithiddendim, 3), nn.Softmax(dim=-1))
            self.head_attri3 = nn.Sequential(nn.Linear(vithiddendim, 3), nn.Softmax(dim=-1))
            self.head_attri4 = nn.Sequential(nn.Linear(vithiddendim, 3), nn.Softmax(dim=-1))
            self.head_attri5 = nn.Sequential(nn.Linear(vithiddendim, 3), nn.Softmax(dim=-1))
            self.head_attri6 = nn.Sequential(nn.Linear(vithiddendim, 2), nn.Softmax(dim=-1))
        if self.numcaps == 6:
            self.head_attri0 = nn.Linear(vithiddendim, 1)
            self.head_attri1 = nn.Linear(vithiddendim, 1)
            self.head_attri2 = nn.Linear(vithiddendim, 1)
            self.head_attri3 = nn.Linear(vithiddendim, 1)
            self.head_attri4 = nn.Linear(vithiddendim, 1)
            self.head_attri5 = nn.Linear(vithiddendim, 1)

        self.encoder_tar = Encoder(seq_length=self.vit.seq_length, num_layers=1, num_heads=num_heads, hidden_dim=vithiddendim,
                                mlp_dim=vitmlpdim,dropout=vitdropout,attention_dropout=vitattentiondropout,norm_layer=self.vit.norm_layer)

        if self.numcaps == 7:
            self.head_tar = nn.Sequential(
                nn.Linear(vithiddendim, 5),
                nn.Softmax(dim=-1)
            )
        else:
            self.head_tar = nn.Linear(vithiddendim, 1)
        self.fc8to1 = nn.Linear(self.numcaps,1)

        decoderlayers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(12):
            decoderlayers[f"decoder_layer_{i}"] = DecoderBlock(
                num_heads,
                vithiddendim,
                vitmlpdim,
                self.vit.dropout,
                vitattentiondropout,
                self.vit.norm_layer,
            )
        self.decoderlayers = nn.Sequential(decoderlayers)
        self.decoder_decon = nn.ConvTranspose2d(vithiddendim,vithiddendim,kernel_size=(16,16),stride=(16,16))
        self.decoder_norm = nn.BatchNorm2d(num_features=vithiddendim)
        self.decoder_sigm = nn.Sigmoid()

        # Prototype vectors
        if args.dataset == "derm7pt":
            self.protodigis0 = nn.Parameter(torch.rand((3, args.num_protos, 197, vithiddendim)),
                                            requires_grad=True)
            self.protodigis1 = nn.Parameter(torch.rand((2, args.num_protos, 197, vithiddendim)),
                                            requires_grad=True)
            self.protodigis2 = nn.Parameter(torch.rand((3, args.num_protos, 197, vithiddendim)),
                                            requires_grad=True)
            self.protodigis3 = nn.Parameter(torch.rand((3, args.num_protos, 197, vithiddendim)),
                                            requires_grad=True)
            self.protodigis4 = nn.Parameter(torch.rand((3, args.num_protos, 197, vithiddendim)),
                                            requires_grad=True)
            self.protodigis5 = nn.Parameter(torch.rand((3, args.num_protos, 197, vithiddendim)),
                                            requires_grad=True)
            self.protodigis6 = nn.Parameter(torch.rand((2, args.num_protos, 197, vithiddendim)),
                                            requires_grad=True)
            self.protodigis_list = [self.protodigis0, self.protodigis1, self.protodigis2, self.protodigis3,
                                    self.protodigis4, self.protodigis5, self.protodigis6]
        if args.dataset == "LIDC":
            self.protodigis0 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis1 = nn.Parameter(torch.rand((4, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis2 = nn.Parameter(torch.rand((6, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis3 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis4 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis5 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis6 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis7 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis_list = [self.protodigis0, self.protodigis1, self.protodigis2, self.protodigis3,
                                self.protodigis4, self.protodigis5, self.protodigis6, self.protodigis7]
        if args.dataset == "FunnyNodules":
            self.protodigis0 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis1 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis2 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis3 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis4 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis5 = nn.Parameter(torch.rand((5, args.num_protos, 197, vithiddendim)), requires_grad=True)
            self.protodigis_list = [self.protodigis0, self.protodigis1, self.protodigis2, self.protodigis3,
                                self.protodigis4, self.protodigis5]

    def attribute_lv(self, input):
        if self.numcaps == 7:
            processed_ = self.vit._process_input(input)
        else:
            processed_ = self.vit._process_input(input.repeat(1, 3, 1, 1))
        batch_size = processed_.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([batch_class_token, processed_], dim=1)
        features = self.vit.encoder(all_tokens)

        if self.numcaps == 7:
            features_attri0 = self.encoder_attri0(features)
            features_attri1 = self.encoder_attri1(features)
            features_attri2 = self.encoder_attri2(features)
            features_attri3 = self.encoder_attri3(features)
            features_attri4 = self.encoder_attri4(features)
            features_attri5 = self.encoder_attri5(features)
            features_attri6 = self.encoder_attri6(features)
        if self.numcaps == 8:
            features_attri0 = self.encoder_attri0(features)
            features_attri1 = self.encoder_attri1(features)
            features_attri2 = self.encoder_attri2(features)
            features_attri3 = self.encoder_attri3(features)
            features_attri4 = self.encoder_attri4(features)
            features_attri5 = self.encoder_attri5(features)
            features_attri6 = self.encoder_attri6(features)
            features_attri7 = self.encoder_attri7(features)
        if self.numcaps == 6:
            features_attri0 = self.encoder_attri0(features)
            features_attri1 = self.encoder_attri1(features)
            features_attri2 = self.encoder_attri2(features)
            features_attri3 = self.encoder_attri3(features)
            features_attri4 = self.encoder_attri4(features)
            features_attri5 = self.encoder_attri5(features)

        if self.numcaps == 7:
            stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                 features_attri4, features_attri5, features_attri6], dim=1)
        elif self.numcaps == 8:
            stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                 features_attri4, features_attri5, features_attri6, features_attri7],
                                                dim=1)
        elif self.numcaps == 6:
            stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                 features_attri4, features_attri5],
                                                dim=1)
        return stacked_attrifeatures


    def forward(self, input):
        if self.numcaps == 7:
            processed_ = self.vit._process_input(input)
        else:
            processed_ = self.vit._process_input(input.repeat(1, 3, 1, 1))
        batch_size = processed_.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([batch_class_token, processed_], dim=1)
        features = self.vit.encoder(all_tokens)

        if self.numcaps in [8,6]:

            decoder_out = self.decoderlayers(features[:,1:])
            reshaped_decoder_out = torch.reshape(decoder_out,(decoder_out.shape[0],14,14,decoder_out.shape[2]))
            permuted_decoder_out = torch.permute(reshaped_decoder_out,(0,3,1,2))
            decoder_tconv = self.decoder_decon(permuted_decoder_out)
            decoder_tconv = self.decoder_norm(decoder_tconv)
            decoder_tconv = torch.permute(decoder_tconv,(0,2,3,1))
            decoder_tconv = self.decoder_sigm(decoder_tconv)
            decoder_tconv = torch.mean(decoder_tconv,dim=-1)
        else:
            decoder_tconv = 0

        if self.numcaps == 7:
            features_attri0 = self.encoder_attri0(features)
            features_attri1 = self.encoder_attri1(features)
            features_attri2 = self.encoder_attri2(features)
            features_attri3 = self.encoder_attri3(features)
            features_attri4 = self.encoder_attri4(features)
            features_attri5 = self.encoder_attri5(features)
            features_attri6 = self.encoder_attri6(features)
        if self.numcaps == 8:
            features_attri0 = self.encoder_attri0(features)
            features_attri1 = self.encoder_attri1(features)
            features_attri2 = self.encoder_attri2(features)
            features_attri3 = self.encoder_attri3(features)
            features_attri4 = self.encoder_attri4(features)
            features_attri5 = self.encoder_attri5(features)
            features_attri6 = self.encoder_attri6(features)
            features_attri7 = self.encoder_attri7(features)
        if self.numcaps == 6:
            features_attri0 = self.encoder_attri0(features)
            features_attri1 = self.encoder_attri1(features)
            features_attri2 = self.encoder_attri2(features)
            features_attri3 = self.encoder_attri3(features)
            features_attri4 = self.encoder_attri4(features)
            features_attri5 = self.encoder_attri5(features)
        if self.numcaps == 7:
            pred_attri0 = self.head_attri0(features_attri0[:,0])
            pred_attri1 = self.head_attri1(features_attri1[:,0])
            pred_attri2 = self.head_attri2(features_attri2[:,0])
            pred_attri3 = self.head_attri3(features_attri3[:,0])
            pred_attri4 = self.head_attri4(features_attri4[:,0])
            pred_attri5 = self.head_attri5(features_attri5[:,0])
            pred_attri6 = self.head_attri6(features_attri6[:,0])
        if self.numcaps == 8:
            pred_attri0 = self.head_attri0(features_attri0[:,0])
            pred_attri1 = self.head_attri1(features_attri1[:,0])
            pred_attri2 = self.head_attri2(features_attri2[:,0])
            pred_attri3 = self.head_attri3(features_attri3[:,0])
            pred_attri4 = self.head_attri4(features_attri4[:,0])
            pred_attri5 = self.head_attri5(features_attri5[:,0])
            pred_attri6 = self.head_attri6(features_attri6[:,0])
            pred_attri7 = self.head_attri7(features_attri7[:,0])
        if self.numcaps == 6:
            pred_attri0 = self.head_attri0(features_attri0[:,0])
            pred_attri1 = self.head_attri1(features_attri1[:,0])
            pred_attri2 = self.head_attri2(features_attri2[:,0])
            pred_attri3 = self.head_attri3(features_attri3[:,0])
            pred_attri4 = self.head_attri4(features_attri4[:,0])
            pred_attri5 = self.head_attri5(features_attri5[:,0])

        if self.numcaps == 7:
            stacked_attrifeatures = torch.stack([features_attri0,features_attri1,features_attri2,features_attri3,
                                                 features_attri4,features_attri5,features_attri6], dim=1)
        elif self.numcaps == 8:
            stacked_attrifeatures = torch.stack([features_attri0,features_attri1,features_attri2,features_attri3,
                                                 features_attri4,features_attri5,features_attri6,features_attri7], dim=1)
        elif self.numcaps == 6:
            stacked_attrifeatures = torch.stack([features_attri0,features_attri1,features_attri2,features_attri3,
                                                 features_attri4,features_attri5], dim=1)

        attriencoderouts_to_encoderin = torch.squeeze(self.fc8to1(stacked_attrifeatures.permute(0,2,3,1)))
        if len(attriencoderouts_to_encoderin.shape) < 3:
            attriencoderouts_to_encoderin = torch.unsqueeze(attriencoderouts_to_encoderin, dim=0)
        features_tar = self.encoder_tar(attriencoderouts_to_encoderin)

        pred_tar = self.head_tar(features_tar[:, 0])

        if self.numcaps == 7:
            x = torch.cat(
                [pred_attri0, pred_attri1, pred_attri2, pred_attri3, pred_attri4, pred_attri5, pred_attri6, pred_tar], dim=-1)
        elif self.numcaps == 8:
            x = torch.cat(
                [pred_attri0, pred_attri1, pred_attri2, pred_attri3, pred_attri4, pred_attri5,
                 pred_attri6, pred_attri7, pred_tar], dim=-1)
        elif self.numcaps == 6:
            x = torch.cat(
                [pred_attri0, pred_attri1, pred_attri2, pred_attri3, pred_attri4, pred_attri5,
                 pred_tar], dim=-1)

        return x, decoder_tconv

    def getDistance(self, input):
        """
        Capsule wise calculation of distance to closest prototype vector
        :param x: vectors to calculate distance to
        :return: distances to closest protoype vector
        """
        if self.numcaps == 7:
            processed_ = self.vit._process_input(input)
        else:
            processed_ = self.vit._process_input(input.repeat(1, 3, 1, 1))
        batch_size = processed_.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([batch_class_token, processed_], dim=1)
        features = self.vit.encoder(all_tokens)

        if self.numcaps == 7:
            features_attri0 = self.encoder_attri0(features)
            features_attri1 = self.encoder_attri1(features)
            features_attri2 = self.encoder_attri2(features)
            features_attri3 = self.encoder_attri3(features)
            features_attri4 = self.encoder_attri4(features)
            features_attri5 = self.encoder_attri5(features)
            features_attri6 = self.encoder_attri6(features)
        if self.numcaps == 8:
            features_attri0 = self.encoder_attri0(features)
            features_attri1 = self.encoder_attri1(features)
            features_attri2 = self.encoder_attri2(features)
            features_attri3 = self.encoder_attri3(features)
            features_attri4 = self.encoder_attri4(features)
            features_attri5 = self.encoder_attri5(features)
            features_attri6 = self.encoder_attri6(features)
            features_attri7 = self.encoder_attri7(features)
        if self.numcaps == 6:
            features_attri0 = self.encoder_attri0(features)
            features_attri1 = self.encoder_attri1(features)
            features_attri2 = self.encoder_attri2(features)
            features_attri3 = self.encoder_attri3(features)
            features_attri4 = self.encoder_attri4(features)
            features_attri5 = self.encoder_attri5(features)
        if self.numcaps == 8:
            x = torch.stack([features_attri0.flatten(start_dim=-2), features_attri1.flatten(start_dim=-2), features_attri2.flatten(start_dim=-2), features_attri3.flatten(start_dim=-2),
                 features_attri4.flatten(start_dim=-2), features_attri5.flatten(start_dim=-2), features_attri6.flatten(start_dim=-2), features_attri7.flatten(start_dim=-2)], dim=1)
        elif self.numcaps == 7:
            x = torch.stack([features_attri0.flatten(start_dim=-2), features_attri1.flatten(start_dim=-2), features_attri2.flatten(start_dim=-2), features_attri3.flatten(start_dim=-2),
                 features_attri4.flatten(start_dim=-2), features_attri5.flatten(start_dim=-2), features_attri6.flatten(start_dim=-2)], dim=1)
        elif self.numcaps == 6:
            x = torch.stack([features_attri0.flatten(start_dim=-2), features_attri1.flatten(start_dim=-2), features_attri2.flatten(start_dim=-2), features_attri3.flatten(start_dim=-2),
                 features_attri4.flatten(start_dim=-2), features_attri5.flatten(start_dim=-2)], dim=1)
        xreshaped = torch.unsqueeze(x, dim=1)
        xreshaped = torch.unsqueeze(xreshaped, dim=1)
        protoreshaped_list = []
        for i in range(len(self.protodigis_list)):
            protoreshaped_list.append(torch.unsqueeze(self.protodigis_list[i], dim=0).flatten(start_dim=-2))
        dists_to_protos = []
        for i in range(len(self.protodigis_list)):
            dists_to_protos.append((xreshaped[:, :, :, i, :] - protoreshaped_list[i]).pow(2).sum(-1).sqrt())
        return dists_to_protos

    def forwardprotodigis(self, x_ex):

        if self.numcaps == 8:
            features_attri0 = x_ex[:,0]
            features_attri1 = x_ex[:,1]
            features_attri2 = x_ex[:,2]
            features_attri3 = x_ex[:,3]
            features_attri4 = x_ex[:,4]
            features_attri5 = x_ex[:,5]
            features_attri6 = x_ex[:,6]
            features_attri7 = x_ex[:,7]
        elif self.numcaps == 7:
            features_attri0 = x_ex[:,0]
            features_attri1 = x_ex[:,1]
            features_attri2 = x_ex[:,2]
            features_attri3 = x_ex[:,3]
            features_attri4 = x_ex[:,4]
            features_attri5 = x_ex[:,5]
            features_attri6 = x_ex[:,6]
        elif self.numcaps == 6:
            features_attri0 = x_ex[:,0]
            features_attri1 = x_ex[:,1]
            features_attri2 = x_ex[:,2]
            features_attri3 = x_ex[:,3]
            features_attri4 = x_ex[:,4]
            features_attri5 = x_ex[:,5]

        if self.numcaps == 8:
            stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                 features_attri4, features_attri5, features_attri6, features_attri7], dim=1)
        elif self.numcaps == 7:
            stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                 features_attri4, features_attri5, features_attri6], dim=1)
        elif self.numcaps == 6:
            stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                 features_attri4, features_attri5], dim=1)

        attriencoderouts_to_encoderin = torch.squeeze(self.fc8to1(stacked_attrifeatures.permute(0, 2, 3, 1)))
        if len(attriencoderouts_to_encoderin.shape) < 3:
            attriencoderouts_to_encoderin = torch.unsqueeze(attriencoderouts_to_encoderin, dim=0)
        features_tar = self.encoder_tar(attriencoderouts_to_encoderin)
        pred_tar = self.head_tar(features_tar[:, 0])

        return pred_tar


    def forward_plot(self, input):
        if self.numcaps == 7:
            processed_ = self.vit._process_input(input)
        else:
            processed_ = self.vit._process_input(input.repeat(1, 3, 1, 1))
        batch_size = processed_.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([batch_class_token, processed_], dim=1)
        features = self.vit.encoder(all_tokens)

        if self.numcaps in [8,6]:
            decoder_out = self.decoderlayers(features[:,1:])
            reshaped_decoder_out = torch.reshape(decoder_out,(decoder_out.shape[0],14,14,decoder_out.shape[2]))
            permuted_decoder_out = torch.permute(reshaped_decoder_out,(0,3,1,2))
            decoder_tconv = self.decoder_decon(permuted_decoder_out)
            decoder_tconv = self.decoder_norm(decoder_tconv)
            decoder_tconv = torch.permute(decoder_tconv,(0,2,3,1))
            decoder_tconv = self.decoder_sigm(decoder_tconv)
            decoder_tconv = torch.mean(decoder_tconv,dim=-1)
        else:
            decoder_tconv = 0

        features_attri0 = self.encoder_attri0(features)
        features_attri1 = self.encoder_attri1(features)
        features_attri2 = self.encoder_attri2(features)
        features_attri3 = self.encoder_attri3(features)
        features_attri4 = self.encoder_attri4(features)
        features_attri5 = self.encoder_attri5(features)
        if self.numcaps == 7:
            features_attri6 = self.encoder_attri6(features)
        if self.numcaps == 8:
            features_attri6 = self.encoder_attri6(features)
            features_attri7 = self.encoder_attri7(features)

        pred_attri0 = self.head_attri0(features_attri0[:,0])
        pred_attri1 = self.head_attri1(features_attri1[:,0])
        pred_attri2 = self.head_attri2(features_attri2[:,0])
        pred_attri3 = self.head_attri3(features_attri3[:,0])
        pred_attri4 = self.head_attri4(features_attri4[:,0])
        pred_attri5 = self.head_attri5(features_attri5[:,0])
        if self.numcaps == 7:
            pred_attri6 = self.head_attri6(features_attri6[:,0])
            stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                 features_attri4, features_attri5, features_attri6],
                                                dim=1)
        elif self.numcaps == 8:
            pred_attri6 = self.head_attri6(features_attri6[:,0])
            pred_attri7 = self.head_attri7(features_attri7[:,0])
            stacked_attrifeatures = torch.stack([features_attri0,features_attri1,features_attri2,features_attri3,
                                                 features_attri4,features_attri5,features_attri6,features_attri7], dim=1)
        else:
            stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                 features_attri4, features_attri5],
                                                dim=1)

        attriencoderouts_to_encoderin = torch.squeeze(self.fc8to1(stacked_attrifeatures.permute(0,2,3,1)))
        if len(attriencoderouts_to_encoderin.shape) < 3:
            attriencoderouts_to_encoderin = torch.unsqueeze(attriencoderouts_to_encoderin, dim=0)
        features_tar = self.encoder_tar(attriencoderouts_to_encoderin)
        pred_tar = self.head_tar(features_tar[:, 0])

        if self.numcaps == 8:
            x = torch.cat(
                    [pred_attri0, pred_attri1, pred_attri2, pred_attri3, pred_attri4, pred_attri5,
                     pred_attri6, pred_attri7, pred_tar], dim=-1)
        elif self.numcaps==7:
            x = torch.cat(
                    [pred_attri0, pred_attri1, pred_attri2, pred_attri3, pred_attri4, pred_attri5,
                     pred_attri6, pred_tar], dim=-1)
        elif self.numcaps == 6:
            x = torch.cat(
                    [pred_attri0, pred_attri1, pred_attri2, pred_attri3, pred_attri4, pred_attri5,
                     pred_tar], dim=-1)

        all_samples_list_vit_attention = []
        for i in range(pred_tar.shape[0]):
            torch.cuda.empty_cache()
            inputto_attri0 = self.encoder_attri0.layers.encoder_layer_0.ln_1(self.encoder_attri0.dropout(all_tokens + self.vit.encoder.pos_embedding))
            attri0_selfattention = self.encoder_attri0.layers.encoder_layer_0.self_attention(inputto_attri0,inputto_attri0,inputto_attri0,need_weights=False)
            vit_attention_attri0 = torch.sum(attri0_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()
            inputto_attri1 = self.encoder_attri1.layers.encoder_layer_0.ln_1(self.encoder_attri1.dropout(all_tokens + self.vit.encoder.pos_embedding))
            attri1_selfattention = self.encoder_attri1.layers.encoder_layer_0.self_attention(inputto_attri1,inputto_attri1,inputto_attri1,need_weights=False)
            vit_attention_attri1 = torch.sum(attri1_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()
            inputto_attri2 = self.encoder_attri2.layers.encoder_layer_0.ln_1(self.encoder_attri2.dropout(all_tokens + self.vit.encoder.pos_embedding))
            attri2_selfattention = self.encoder_attri2.layers.encoder_layer_0.self_attention(inputto_attri2,inputto_attri2,inputto_attri2,need_weights=False)
            vit_attention_attri2 = torch.sum(attri2_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()
            inputto_attri3 = self.encoder_attri3.layers.encoder_layer_0.ln_1(self.encoder_attri3.dropout(all_tokens + self.vit.encoder.pos_embedding))
            attri3_selfattention = self.encoder_attri3.layers.encoder_layer_0.self_attention(inputto_attri3,inputto_attri3,inputto_attri3,need_weights=False)
            vit_attention_attri3 = torch.sum(attri3_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()
            inputto_attri4 = self.encoder_attri4.layers.encoder_layer_0.ln_1(self.encoder_attri4.dropout(all_tokens + self.vit.encoder.pos_embedding))
            attri4_selfattention = self.encoder_attri4.layers.encoder_layer_0.self_attention(inputto_attri4,inputto_attri4,inputto_attri4,need_weights=False)
            vit_attention_attri4 = torch.sum(attri4_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()
            inputto_attri5 = self.encoder_attri5.layers.encoder_layer_0.ln_1(self.encoder_attri5.dropout(all_tokens + self.vit.encoder.pos_embedding))
            attri5_selfattention = self.encoder_attri5.layers.encoder_layer_0.self_attention(inputto_attri5,inputto_attri5,inputto_attri5,need_weights=False)
            vit_attention_attri5 = torch.sum(attri5_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()
            if self.numcaps == 7:
                inputto_attri6 = self.encoder_attri6.layers.encoder_layer_0.ln_1(self.encoder_attri6.dropout(all_tokens + self.vit.encoder.pos_embedding))
                attri6_selfattention = self.encoder_attri6.layers.encoder_layer_0.self_attention(inputto_attri6,inputto_attri6,inputto_attri6,need_weights=False)
                vit_attention_attri6 = torch.sum(attri6_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()
                stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                     features_attri4, features_attri5, features_attri6], dim=1)
            elif self.numcaps == 8:
                inputto_attri6 = self.encoder_attri6.layers.encoder_layer_0.ln_1(self.encoder_attri6.dropout(all_tokens + self.vit.encoder.pos_embedding))
                attri6_selfattention = self.encoder_attri6.layers.encoder_layer_0.self_attention(inputto_attri6,inputto_attri6,inputto_attri6,need_weights=False)
                vit_attention_attri6 = torch.sum(attri6_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()
                inputto_attri7 = self.encoder_attri7.layers.encoder_layer_0.ln_1(self.encoder_attri7.dropout(all_tokens + self.vit.encoder.pos_embedding))
                attri7_selfattention = self.encoder_attri7.layers.encoder_layer_0.self_attention(inputto_attri7,inputto_attri7,inputto_attri7,need_weights=False)
                vit_attention_attri7 = torch.sum(attri7_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()

                stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                     features_attri4, features_attri5, features_attri6, features_attri7], dim=1)
            else:
                stacked_attrifeatures = torch.stack([features_attri0, features_attri1, features_attri2, features_attri3,
                                                     features_attri4, features_attri5], dim=1)

            attriencoderouts_to_encoderin = torch.squeeze(self.fc8to1(stacked_attrifeatures.permute(0, 2, 3, 1)))
            inputto_tar = self.encoder_tar.layers.encoder_layer_0.ln_1(self.encoder_tar.dropout(attriencoderouts_to_encoderin))
            tar_selfattention = self.encoder_tar.layers.encoder_layer_0.self_attention(inputto_tar, inputto_tar,
                                                                                       inputto_tar, need_weights=False)

            vit_attention_tar = torch.sum(tar_selfattention[0][i], dim=-1)[1:].reshape((14, 14)).cpu().detach().numpy()


            list_vit_attention = []

            list_vit_attention.append(input[i,0].cpu())
            if self.numcaps == 8:
                list_vit_attention.append(torch.squeeze(decoder_tconv[i]).cpu().detach().numpy())
            else:
                list_vit_attention.append(input[i, 0].cpu())
            list_vit_attention.append(vit_attention_attri0)
            list_vit_attention.append(vit_attention_attri1)
            list_vit_attention.append(vit_attention_attri2)
            list_vit_attention.append(vit_attention_attri3)
            list_vit_attention.append(vit_attention_attri4)
            list_vit_attention.append(vit_attention_attri5)
            if self.numcaps == 7:
                list_vit_attention.append(vit_attention_attri6)
            if self.numcaps == 8:
                list_vit_attention.append(vit_attention_attri6)
                list_vit_attention.append(vit_attention_attri7)

            list_vit_attention.append(vit_attention_tar)

            all_samples_list_vit_attention.append(list_vit_attention)

        return x, decoder_tconv, all_samples_list_vit_attention

