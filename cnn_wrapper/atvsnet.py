# -*- coding: utf-8 -*
from cnn_wrapper.network import Network


class StackedUNet(Network):
    def setup(self):
        base_filter = 8
        # block 1
        (self.feed('data')
         .conv_bn(3, base_filter * 2, 2, name='conv_b0_1_0')
         .conv_bn(3, base_filter * 4, 2, name='conv_b0_2_0')
         .conv_bn(3, base_filter * 8, 2, name='conv_b0_3_0'))

        (self.feed('data')
         .conv_bn(3, base_filter, 1, name='conv_b0_0_1'))

        (self.feed('conv_b0_1_0')
         .conv_bn(3, base_filter * 2, 1, name='conv_b0_1_1'))

        (self.feed('conv_b0_2_0')
         .conv_bn(3, base_filter * 4, 1, name='conv_b0_2_1'))

        (self.feed('conv_b0_3_0')
         .conv_bn(3, base_filter * 8, 1, name='conv_b0_3_1')
         .deconv_bn(3, base_filter * 4, 2, name='conv_b0_4_0'))

        (self.feed('conv_b0_4_0', 'conv_b0_2_1')
         .add(name='conv_b0_4_1')
         .deconv_bn(3, base_filter * 2, 2, name='conv_b0_5_0'))

        (self.feed('conv_b0_5_0', 'conv_b0_1_1')
         .add(name='conv_b0_5_1')
         .deconv_bn(3, base_filter, 2, name='conv_b0_6_0'))

        # block 2
        (self.feed('conv_b0_6_0', 'conv_b0_0_1')
         .add(name='conv_b1_0_0')
         .conv_bn(3, base_filter * 2, 2, name='conv_b1_1_0')
         .conv_bn(3, base_filter * 4, 2, name='conv_b1_2_0')
         .conv_bn(3, base_filter * 8, 2, name='conv_b1_3_0'))

        (self.feed('conv_b1_0_0')
         .conv_bn(3, base_filter, 1, name='conv_b1_0_1'))

        (self.feed('conv_b1_1_0', 'conv_b0_5_0')
         .add(name='conv_b1_1_1_concat')
         .conv_bn(3, base_filter * 2, 1, name='conv_b1_1_1'))

        (self.feed('conv_b1_2_0', 'conv_b0_4_0')
         .add(name='conv_b1_2_1_concat')
         .conv_bn(3, base_filter * 4, 1, name='conv_b1_2_1'))

        (self.feed('conv_b1_3_0')
         .conv_bn(3, base_filter * 8, 1, name='conv_b1_3_1')
         .deconv_bn(3, base_filter * 4, 2, name='conv_b1_4_0'))

        (self.feed('conv_b1_4_0', 'conv_b1_2_1', 'conv_b0_2_1')
         .add(name='conv_b1_4_1')
         .deconv_bn(3, base_filter * 2, 2, name='conv_b1_5_0'))

        (self.feed('conv_b1_5_0', 'conv_b1_1_1', 'conv_b0_1_1')
         .add(name='conv_b1_5_1')
         .deconv_bn(3, base_filter, 2, name='conv_b1_6_0'))

        # block 3
        (self.feed('conv_b1_6_0', 'conv_b1_0_1')
         .add(name='conv_b2_0_0')
         .conv_bn(3, base_filter * 2, 2, name='conv_b2_1_0')
         .conv_bn(3, base_filter * 4, 2, name='conv_b2_2_0')
         .conv_bn(3, base_filter * 8, 2, name='conv_b2_3_0'))

        (self.feed('conv_b2_0_0')
         .conv_bn(3, base_filter, 1, name='conv_b2_0_1'))

        (self.feed('conv_b2_1_0', 'conv_b1_5_0')
         .add(name='conv_b2_1_1_concat')
         .conv_bn(3, base_filter * 2, 1, name='conv_b2_1_1'))

        (self.feed('conv_b2_2_0', 'conv_b1_4_0')
         .add(name='conv_b2_2_1_concat')
         .conv_bn(3, base_filter * 4, 1, name='conv_b2_2_1'))

        (self.feed('conv_b2_3_0')
         .conv_bn(3, base_filter * 8, 1, name='conv_b2_3_1')
         .deconv_bn(3, base_filter * 4, 2, name='conv_b2_4_0'))

        (self.feed('conv_b2_4_0', 'conv_b2_2_1', 'conv_b0_2_1')
         .add(name='conv_b2_4_1')
         .deconv_bn(3, base_filter * 2, 2, name='conv_b2_5_0'))

        (self.feed('conv_b2_5_0', 'conv_b2_1_1', 'conv_b0_1_1')
         .add(name='conv_b2_5_1')
         .deconv_bn(3, base_filter, 2, name='conv_b2_6_0'))

        (self.feed('conv_b2_6_0', 'conv_b2_0_1')
         .add(name='conv_b2_6_1'))


# StackedUNet with prob volume output
class StackedUNet_prob(Network):
    def setup(self):
        base_filter = 8        
        # block 1
        (self.feed('data')
        .conv_bn(3, base_filter * 2, 2, name='conv_b0_1_0')
        .conv_bn(3, base_filter * 4, 2, name='conv_b0_2_0')
        .conv_bn(3, base_filter * 8, 2, name='conv_b0_3_0'))

        (self.feed('data')
        .conv_bn(3, base_filter, 1, name='conv_b0_0_1'))

        (self.feed('conv_b0_1_0')
        .conv_bn(3, base_filter * 2, 1, name='conv_b0_1_1'))

        (self.feed('conv_b0_2_0')
        .conv_bn(3, base_filter * 4, 1, name='conv_b0_2_1'))

        (self.feed('conv_b0_3_0')
        .conv_bn(3, base_filter * 8, 1, name='conv_b0_3_1')
        .deconv_bn(3, base_filter * 4, 2, name='conv_b0_4_0'))

        (self.feed('conv_b0_4_0', 'conv_b0_2_1')
        .add(name='conv_b0_4_1')
        .deconv_bn(3, base_filter * 2, 2, name='conv_b0_5_0'))

        (self.feed('conv_b0_5_0', 'conv_b0_1_1')
        .add(name='conv_b0_5_1')
        .deconv_bn(3, base_filter, 2, name='conv_b0_6_0'))

        # block 2
        (self.feed('conv_b0_6_0', 'conv_b0_0_1')
        .add(name='conv_b1_0_0')
        .conv_bn(3, base_filter * 2, 2, name='conv_b1_1_0')
        .conv_bn(3, base_filter * 4, 2, name='conv_b1_2_0')
        .conv_bn(3, base_filter * 8, 2, name='conv_b1_3_0'))

        (self.feed('conv_b1_0_0')
        .conv_bn(3, base_filter, 1, name='conv_b1_0_1'))

        (self.feed('conv_b1_1_0', 'conv_b0_5_0')
        .add(name='conv_b1_1_1_concat')
        .conv_bn(3, base_filter * 2, 1, name='conv_b1_1_1'))

        (self.feed('conv_b1_2_0', 'conv_b0_4_0')
        .add(name='conv_b1_2_1_concat')
        .conv_bn(3, base_filter * 4, 1, name='conv_b1_2_1'))

        (self.feed('conv_b1_3_0')
        .conv_bn(3, base_filter * 8, 1, name='conv_b1_3_1')
        .deconv_bn(3, base_filter * 4, 2, name='conv_b1_4_0'))

        (self.feed('conv_b1_4_0', 'conv_b1_2_1', 'conv_b0_2_1')
        .add(name='conv_b1_4_1')
        .deconv_bn(3, base_filter * 2, 2, name='conv_b1_5_0'))

        (self.feed('conv_b1_5_0', 'conv_b1_1_1', 'conv_b0_1_1')
        .add(name='conv_b1_5_1')
        .deconv_bn(3, base_filter, 2, name='conv_b1_6_0'))

        # block 3
        (self.feed('conv_b1_6_0', 'conv_b1_0_1')
        .add(name='conv_b2_0_0')
        .conv_bn(3, base_filter * 2, 2, name='conv_b2_1_0')
        .conv_bn(3, base_filter * 4, 2, name='conv_b2_2_0')
        .conv_bn(3, base_filter * 8, 2, name='conv_b2_3_0'))

        (self.feed('conv_b2_0_0')
        .conv_bn(3, base_filter, 1, name='conv_b2_0_1'))

        (self.feed('conv_b2_1_0', 'conv_b1_5_0')
        .add(name='conv_b2_1_1_concat')
        .conv_bn(3, base_filter * 2, 1, name='conv_b2_1_1'))

        (self.feed('conv_b2_2_0', 'conv_b1_4_0')
        .add(name='conv_b2_2_1_concat')
        .conv_bn(3, base_filter * 4, 1, name='conv_b2_2_1'))

        (self.feed('conv_b2_3_0')
        .conv_bn(3, base_filter * 8, 1, name='conv_b2_3_1')
        .deconv_bn(3, base_filter * 4, 2, name='conv_b2_4_0'))

        (self.feed('conv_b2_4_0', 'conv_b2_2_1', 'conv_b0_2_1')
        .add(name='conv_b2_4_1')
        .deconv_bn(3, base_filter * 2, 2, name='conv_b2_5_0'))

        (self.feed('conv_b2_5_0', 'conv_b2_1_1', 'conv_b0_1_1')
        .add(name='conv_b2_5_1')
        .deconv_bn(3, base_filter, 2, name='conv_b2_6_0'))

        (self.feed('conv_b2_6_0', 'conv_b2_0_1')
        .add(name='conv_b2_6_1')
        .conv(3, 1, 1, relu=False, name='conv_b2_6_2'))



class AttAggregation_keepchannel(Network):
    def setup(self):
        base_filter = 8
        # data size of (B, D, H, W, C, N), N=NumNeigh
        shape_in = self.get_shape_by_name('data')
        (self.feed('data')
         .attention_aggregation(kernel_size=3, name='attention_aggregate', second_weight=True, relu=True, biased=False, n_view=shape_in[-1])
        )


class AttAggregation(Network):
    def setup(self):
        base_filter = 8
        # data size of (B, D, H, W, C, N), N=NumNeigh
        shape_in = self.get_shape_by_name('data')
        (self.feed('data')
         .attention_aggregation(kernel_size=3, name='attention_aggregate', second_weight=True, relu=True, biased=False, n_view=shape_in[-1])
         .conv(3, 1, 1, relu=False, name='attention_prob_vol'))


class OutputConv(Network):
    # Conv to set ouput channel to 1 
    def setup(self):
        (self.feed('data')
         .conv(3, 1, 1, relu=False, name='attention_prob_vol'))

class OutputConv_refine(Network):
    # Conv to set ouput channel to 1 
    def setup(self):
        (self.feed('data')
         .conv(3, 1, 1, relu=False, name='attention_prob_vol_refine'))


class AttAggregation_refine_keepchannel(Network):
    def setup(self):
        # data size of (B, D, H, W, C, N), N=NumNeigh
        shape_in = self.get_shape_by_name('data')
        (self.feed('data')
         .attention_aggregation(kernel_size=3, name='attention_aggregate_refine', second_weight=True, relu=True, biased=False, n_view=shape_in[-1]))

class AttAggregation_refine(Network):
    def setup(self):
        # data size of (B, D, H, W, C, N), N=NumNeigh
        shape_in = self.get_shape_by_name('data')
        (self.feed('data')
         .attention_aggregation(kernel_size=3, name='attention_aggregate_refine', second_weight=True, relu=True, biased=False, n_view=shape_in[-1])
         .conv(3, 1, 1, relu=False, name='attention_prob_vol_refine'))


class ResNetDS2SPP_shallow_f16(Network):
    def setup(self):
        base_filter = 16
        # resnet blocks
        (self.feed('data')
         .res_block(3, base_filter, num_block=3, stride=4, rate=1, name='global_refine_conv0_x')
         .conv(1, base_filter, 1, relu=False, name='global_refine_shallow_feature'))


class ResNetDS2SPP(Network):
    def setup(self):
        base_filter = 32

        # resnet blocks
        (self.feed('data')
         .conv_bn(3, base_filter, 2, name='conv0_0')
         .conv_bn(3, base_filter, 1, name='conv0_1')
         .conv_bn(3, base_filter, 1, name='conv0_2')
         .res_block(3, base_filter, num_block=3, stride=1, rate=1, name='conv0_x')
         .res_block(3, base_filter*2, num_block=8, stride=2, rate=1, name='conv1_x')
         .res_block(3, base_filter*4, num_block=3, stride=1, rate=2, name='conv2_x')
         .res_block(3, base_filter*4, num_block=3, stride=1, rate=4, name='conv3_x'))

        # Spatial Pyramid Pooling module
        feature_map_size = self.get_shape_by_name('conv3_x')
        upsample_method = 'bilinear'
        (self.feed('conv3_x')
         .avg_pool(64, 64, name='branch_0_pool')
         .conv_bn(3, base_filter, 1, relu=True, name='branch_0_conv')
         .image_resize(size=(feature_map_size[1], feature_map_size[2]), method=upsample_method, name='branch_0', align_corners=True))
        (self.feed('conv3_x')
         .avg_pool(32, 32, name='branch_1_pool')
         .conv_bn(3, base_filter, 1, relu=True, name='branch_1_conv')
         .image_resize(size=(feature_map_size[1], feature_map_size[2]), method=upsample_method, name='branch_1', align_corners=True))
        (self.feed('conv3_x')
         .avg_pool(16, 16, name='branch_2_pool')
         .conv_bn(3, base_filter, 1, relu=True, name='branch_2_conv')
         .image_resize(size=(feature_map_size[1], feature_map_size[2]), method=upsample_method, name='branch_2', align_corners=True))
        (self.feed('conv3_x')
         .avg_pool(8, 8, name='branch_3_pool')
         .conv_bn(3, base_filter, 1, relu=True, name='branch_3_conv')
         .image_resize(size=(feature_map_size[1], feature_map_size[2]), method=upsample_method, name='branch_3', align_corners=True))

        # Fusion
        (self.feed('conv1_x', 'conv3_x', 'branch_0', 'branch_1', 'branch_2', 'branch_3')
         .concat(axis=-1, name='concat_feature')
         .conv_bn(3, base_filter*4, 1, relu=True, name='fusion0')
         .conv(1, base_filter, 1, relu=False, name='fusion1'))


class CostVolRefineNet(Network):
    def setup(self):
        base_filter = 8

        (self.feed('photo_group')
        .conv_bn(3, base_filter, 1, name='global_refine_photo_3dconv'))
        (self.feed('geo_group')
        .conv_bn(3, base_filter, 1, name='global_refine_geo_3dconv'))
        (self.feed('prob_vol')
        .conv_bn(3, base_filter, 1, name='global_refine_prob_3dconv'))
        (self.feed('vis_hull')
        .conv_bn(3, base_filter, 1, name='global_refine_vishull_3dconv'))

        (self.feed('global_refine_photo_3dconv', 'global_refine_geo_3dconv', 'global_refine_prob_3dconv', 'global_refine_vishull_3dconv')
        .concat(axis=-1, name='global_refine_concat')
        .conv_bn(3, base_filter * 2, 2, name='global_refine_3dconv1_0')
        .conv_bn(3, base_filter * 4, 2, name='global_refine_3dconv2_0')
        .conv_bn(3, base_filter * 8, 2, name='global_refine_3dconv3_0'))

        (self.feed('global_refine_concat')
        .conv_bn(3, base_filter, 1, name='global_refine_3dconv0_1'))
        (self.feed('global_refine_3dconv1_0')
        .conv_bn(3, base_filter * 2, 1, name='global_refine_3dconv1_1'))

        (self.feed('global_refine_3dconv2_0')
        .conv_bn(3, base_filter * 4, 1, name='global_refine_3dconv2_1'))

        (self.feed('global_refine_3dconv3_0')
        .conv_bn(3, base_filter * 8, 1, name='global_refine_3dconv3_1')
        .deconv_bn(3, base_filter * 4, 2, name='global_refine_3dconv4_0'))

        (self.feed('global_refine_3dconv4_0', 'global_refine_3dconv2_1')
        .add(name='global_refine_3dconv4_1')
        .deconv_bn(3, base_filter * 2, 2, name='global_refine_3dconv5_0'))

        (self.feed('global_refine_3dconv5_0', 'global_refine_3dconv1_1')
        .add(name='global_refine_3dconv5_1')
        .deconv_bn(3, base_filter, 2, name='global_refine_3dconv6_0'))

        (self.feed('global_refine_3dconv6_0', 'global_refine_3dconv0_1')
        .add(name='global_refine_3dconv6_1')
        .conv(3, 1, 1, relu=False, name='global_refined_cost_vol'))

