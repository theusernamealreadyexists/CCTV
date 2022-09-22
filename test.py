import torch

checkpoint = torch.load(self.pretrain_path_img, map_location='cuda:0')
self.vae_img_enc.load_state_dict(checkpoint['enc'])