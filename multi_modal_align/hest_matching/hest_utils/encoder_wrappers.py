import timm
import torch
from transformers import ViTModel


class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True):
        super().__init__()
        
        if kwargs.get('pretrained', False) == False:
            # give a warning
            print(f"Warning: {model_name} is used to instantiate a CNN Encoder, but no pretrained weights are loaded. This is expected if you will be loading weights from a checkpoint.")
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
    
    def forward_features(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        return out


class TimmViTEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "vit_large_patch16_224", 
                 kwargs: dict = {'dynamic_img_size': True, 'pretrained': True, 'num_classes': 0}):
        super().__init__()
        
        if kwargs.get('pretrained', False):
            # give a warning
            print(f"Warning: {model_name} is used to instantiate a Timm ViT Encoder, but no pretrained weights are loaded. This is expected if you will be loading weights from a checkpoint.")
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
    
    def forward(self, x):
        out = self.model(x)
        return out
    
    def forward_features(self, x):
        out = self.model.forward_features(x)
        return out


class HFViTEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "owkin/phikon", 
                 kwargs: dict = {'add_pooling_layer': False}):
        super().__init__()
        
        self.model = ViTModel.from_pretrained(model_name, **kwargs)
        self.model_name = model_name
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.last_hidden_state[:, 0, :]
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out
    

class CLIPVisionModelPostProcessor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.pooler_output


class DenseNetBackbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.classifier = torch.nn.Identity()
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.model.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class GigapathSlide(torch.nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()

        # import gigapath.slide_encoder as slide_encoder
        from .gigapath_slide_encoder import create_model

        self.tile_encoder = timm.create_model(model_name='vit_giant_patch14_dinov2', 
                **{'img_size': 224, 'in_chans': 3, 
                'patch_size': 16, 'embed_dim': 1536, 
                'depth': 40, 'num_heads': 24, 'init_values': 1e-05, 
                'mlp_ratio': 5.33334, 'num_classes': 0})
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.tile_encoder.load_state_dict(state_dict, strict=True)

        self.slide_model = create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)

        self.tile_encoder.eval()
        self.slide_model.eval()

    def forward(self, x, coords):
        tile_embed = self.tile_encoder(x)
        _, output = self.slide_model(tile_embed.unsqueeze(0), coords.unsqueeze(0))
        return output[0][0, 1:]


class ConvStem(torch.nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        from timm_ctp.models.layers.helpers import to_2tuple

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(torch.nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(torch.nn.BatchNorm2d(output_dim))
            stem.append(torch.nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(torch.nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = torch.nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def ctranspath(img_size = 224, **kwargs):
    from timm_ctp import create_model as ctp_create_model

    model = ctp_create_model('swin_tiny_patch4_window7_224', 
                                  embed_layer=ConvStem, 
                                  pretrained=False,
                                  img_size=img_size,
                                  **kwargs)
    return model