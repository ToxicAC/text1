import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math
import clip

import gazelle.utils as utils
from gazelle.backbone import DinoV2Backbone


class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.head_token = nn.Embedding(1, self.dim)
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))
        if self.inout: self.inout_token = nn.Embedding(1, self.dim)
        self.transformer = nn.Sequential(*[
            Block(
                dim=self.dim, 
                num_heads=8, 
                mlp_ratio=4, 
                drop_path=0.1)
                for i in range(num_layers)
                ])
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        if self.inout: 
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, input):
        # input["images"]: [B, 3, H, W] tensor of images
        # input["bboxes"]: list of lists of bbox tuples [[(xmin, ymin, xmax, ymax)]] per image in normalized image coords

        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        x = self.backbone.forward(input["images"])
        x = self.linear(x)
        x = x + self.pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img) # repeat image features along people dimension per image
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device) # [sum(N_p), 32, 32]
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :] # slice off inout tokens from scene tokens
        
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img) # resplit per image

        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        # bboxes: [[(xmin, ymin, xmax, ymax)]] - list of list of head bboxes per image
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None: # no bbox provided, use empty head map
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps
    
    def get_gazelle_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}
        
    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set([k for k in keys1 if not k.startswith("backbone")])
            keys2 = set([k for k in keys2 if not k.startswith("backbone")])
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]
        
        self.load_state_dict(current_state_dict, strict=False)


# From https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
    

# models
def get_gazelle_model(model_name):
    factory = {
        "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
        "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
        "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
        "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
    }
    assert model_name in factory.keys(), "invalid model name"
    return factory[model_name]()

def gazelle_dinov2_vitb14():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov2_vitl14():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov2_vitb14_inout():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True)
    return model, transform

def gazelle_dinov2_vitl14_inout():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True)
    return model, transform


# ============================================================
# Prompt-Guided GazeLLE (with CLIP text encoder branch)
# ============================================================

class PromptGazeLLE(nn.Module):
    """GazeLLE with a geometric-aware text prompt branch as training-time teacher.

    During training, the text branch encodes geometric vocabulary from learnable
    anchor points via a frozen CLIP text encoder. The text feature guides the
    vision decoder through an L_MCR alignment loss -- it is NEVER concatenated
    with vision tokens. At inference time, the text branch is completely bypassed.
    """

    def __init__(self, backbone, clip_model_name="ViT-B/32", inout=False,
                 dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        self.num_patches = self.featmap_h * self.featmap_w  # 1024

        # --- CLIP text encoder (frozen, training-only teacher) ---
        clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.clip_dim = clip_model.transformer.width  # 512 for ViT-B/32
        self.clip_text_encoder = clip_model
        for param in self.clip_text_encoder.parameters():
            param.requires_grad = False

        # --- Learnable anchor embeddings for 32x32 grid ---
        self.anchors = nn.Embedding(self.num_patches, self.clip_dim)

        # --- Soft prompts ---
        self.soft_prefix = nn.Parameter(torch.randn(4, self.clip_dim))
        self.soft_mid = nn.Parameter(torch.randn(4, self.clip_dim))

        # --- Text-to-vision projection (for MCR loss space) ---
        self.text_proj = nn.Linear(self.clip_dim, self.dim)

        # --- Vision global feature projection (for MCR loss space) ---
        self.vision_proj = nn.Linear(self.dim, self.dim)

        # --- Vision components (same as GazeLLE) ---
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.head_token = nn.Embedding(1, self.dim)
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)
            .squeeze(dim=0).squeeze(dim=0)
        )
        if self.inout:
            self.inout_token = nn.Embedding(1, self.dim)

        self.transformer = nn.Sequential(*[
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(num_layers)
        ])

        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def encode_text_prompt(self, head_indices, head_weights,
                           target_indices, target_weights):
        """Synthesize geometric vocabulary and encode via frozen CLIP text encoder.

        Args:
            head_indices: [B, 4] LongTensor
            head_weights: [B, 4] FloatTensor
            target_indices: [B, 4] LongTensor
            target_weights: [B, 4] FloatTensor

        Returns:
            f_text: [B, dim] text feature projected to shared space
        """
        B = head_indices.shape[0]

        # Weighted sum of anchor embeddings -> geometric vocabulary
        head_embeds = self.anchors(head_indices)      # [B, 4, clip_dim]
        target_embeds = self.anchors(target_indices)   # [B, 4, clip_dim]

        V_head = (head_embeds * head_weights.unsqueeze(-1)).sum(dim=1, keepdim=True)      # [B, 1, clip_dim]
        V_target = (target_embeds * target_weights.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B, 1, clip_dim]

        # Build text sequence: [soft_prefix(4), V_head(1), soft_mid(4), V_target(1)] = 10 tokens
        prefix = self.soft_prefix.unsqueeze(0).expand(B, -1, -1)  # [B, 4, clip_dim]
        mid = self.soft_mid.unsqueeze(0).expand(B, -1, -1)        # [B, 4, clip_dim]
        text_seq = torch.cat([prefix, V_head, mid, V_target], dim=1)  # [B, 10, clip_dim]

        # Pass through frozen CLIP text encoder
        seq_len = text_seq.size(1)
        x = text_seq + self.clip_text_encoder.positional_embedding[:seq_len]
        x = x.permute(1, 0, 2)  # [seq_len, B, clip_dim]

        # Slice causal attention mask to match actual sequence length
        original_masks = []
        sliced_mask = self.clip_text_encoder.transformer.resblocks[0].attn_mask[:seq_len, :seq_len]
        for block in self.clip_text_encoder.transformer.resblocks:
            original_masks.append(block.attn_mask)
            block.attn_mask = sliced_mask

        x = self.clip_text_encoder.transformer(x)

        # Restore original masks
        for block, orig_mask in zip(self.clip_text_encoder.transformer.resblocks, original_masks):
            block.attn_mask = orig_mask

        x = x.permute(1, 0, 2)  # [B, seq_len, clip_dim]
        x = self.clip_text_encoder.ln_final(x)

        # Extract EOS token feature (last position)
        f_text_raw = x[:, -1, :]  # [B, clip_dim]
        f_text = self.text_proj(f_text_raw)  # [B, dim]
        return f_text

    def forward(self, images, bboxes, target_info=None):
        """
        Args:
            images: [B, 3, H, W] tensor
            bboxes: [[(xmin, ymin, xmax, ymax)]] per image (normalized coords)
            target_info: dict with head_indices, head_weights, target_indices, target_weights
                         (training only, None at inference)

        Returns:
            heatmap_preds: list of [N_i, 64, 64] per image
            inout_preds: list of [N_i] per image (or None if not self.inout)
            f_vision: [sum(N_p), dim] global vision feature (training only)
            f_text: [sum(N_p), dim] text feature (training only)
        """
        num_ppl_per_img = [len(bbox_list) for bbox_list in bboxes]

        # --- Vision branch (always runs) ---
        x = self.backbone.forward(images)
        x = self.linear(x)
        x = x + self.pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img)
        head_maps = torch.cat(
            self.get_input_head_maps(bboxes), dim=0
        ).to(x.device)
        head_map_embeddings = (
            head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        )
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1)  # [sum(N_p), 1024, dim]

        # --- Vision-only ViT decoder (no text token concatenated!) ---
        if self.inout:
            t_in_out = self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
            joint_tokens = torch.cat([t_in_out, x], dim=1)  # [N, 1+1024, dim]
        else:
            joint_tokens = x  # [N, 1024, dim]

        out = self.transformer(joint_tokens)

        # --- In/out prediction ---
        if self.inout:
            inout_preds = self.inout_head(out[:, 0, :]).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            pure_vision = out[:, 1:, :]  # strip inout token
        else:
            inout_preds = None
            pure_vision = out  # all tokens are vision

        # --- Heatmap decoding ---
        x = pure_vision.reshape(
            pure_vision.shape[0], self.featmap_h, self.featmap_w, pure_vision.shape[2]
        ).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        # --- Training-only: compute vision_tokens and f_text for MCR loss ---
        if self.training and target_info is not None:
            # Vision tokens: [sum(N_p), 1024, dim] — keep all spatial tokens
            vision_tokens = self.vision_proj(pure_vision)  # project each token

            # Text feature: encode geometric prompt from ground truth
            head_indices = utils.repeat_tensors(target_info["head_indices"], num_ppl_per_img)
            head_weights = utils.repeat_tensors(target_info["head_weights"], num_ppl_per_img)
            target_indices = utils.repeat_tensors(target_info["target_indices"], num_ppl_per_img)
            target_weights = utils.repeat_tensors(target_info["target_weights"], num_ppl_per_img)
            f_text = self.encode_text_prompt(
                head_indices, head_weights, target_indices, target_weights
            )  # [sum(N_p), dim]

            return heatmap_preds, inout_preds, vision_tokens, f_text

        # Inference: text branch bypassed entirely
        return heatmap_preds, inout_preds, None, None

    def get_input_head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    def get_gazelle_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {
                k: v for k, v in self.state_dict().items()
                if not k.startswith("backbone") and not k.startswith("clip_text_encoder")
            }

    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set([k for k in keys1
                         if not k.startswith("backbone") and not k.startswith("clip_text_encoder")])
            keys2 = set([k for k in keys2
                         if not k.startswith("backbone") and not k.startswith("clip_text_encoder")])
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]

        self.load_state_dict(current_state_dict, strict=False)


# --- PromptGazeLLE factory functions ---

def prompt_gazelle_dinov2_vitb14(clip_model_name="ViT-B/32"):
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = PromptGazeLLE(backbone, clip_model_name=clip_model_name)
    return model, transform

def prompt_gazelle_dinov2_vitl14(clip_model_name="ViT-B/32"):
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = PromptGazeLLE(backbone, clip_model_name=clip_model_name)
    return model, transform

def prompt_gazelle_dinov2_vitb14_inout(clip_model_name="ViT-B/32"):
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = PromptGazeLLE(backbone, clip_model_name=clip_model_name, inout=True)
    return model, transform

def prompt_gazelle_dinov2_vitl14_inout(clip_model_name="ViT-B/32"):
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = PromptGazeLLE(backbone, clip_model_name=clip_model_name, inout=True)
    return model, transform
