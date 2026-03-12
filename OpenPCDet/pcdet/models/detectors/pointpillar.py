from .detector3d_template import Detector3DTemplate
from ..backbones_2d.radar_pillar_attention_block import RadarPillarAttentionBlock


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.radar_pillar_attention_block = RadarPillarAttentionBlock(
            feature_dim=self.map_to_bev_module.num_bev_features
        )

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

            if cur_module.__class__.__name__ == 'PointPillarScatter':
                spatial_features = batch_dict['spatial_features']
                b, c, h, w = spatial_features.shape
                print(f"spatial_features.shape = ({b}, {c}, {h}, {w})")
                batch_dict['spatial_features'] = self.radar_pillar_attention_block(spatial_features)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
