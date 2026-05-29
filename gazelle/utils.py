import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import random
from sklearn.metrics import roc_auc_score

def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)

def split_tensors(tensor, split_counts):
    indices = torch.cumsum(torch.tensor([0] + split_counts), dim=0)
    return [tensor[indices[i]:indices[i+1]] for i in range(len(split_counts))]

def visualize_heatmap(pil_image, heatmap, bbox=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(128)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="green", width=3)
    return overlay_image

def stack_and_pad(tensor_list):
    max_size = max([t.shape[0] for t in tensor_list])
    padded_list = []
    for t in tensor_list:
        if t.shape[0] == max_size:
            padded_list.append(t)
        else:
            padded_list.append(torch.cat([t, torch.zeros(max_size - t.shape[0], *t.shape[1:])], dim=0))
    return torch.stack(padded_list)

def random_crop(img, bbox, gazex, gazey, inout):
    width, height = img.size
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    # determine feasible crop region (must include bbox and gaze target)
    crop_reg_xmin = min(bbox_xmin, min(gazex)) if inout else bbox_xmin
    crop_reg_ymin = min(bbox_ymin, min(gazey)) if inout else bbox_ymin
    crop_reg_xmax = max(bbox_xmax, max(gazex)) if inout else bbox_xmax
    crop_reg_ymax = max(bbox_ymax, max(gazey)) if inout else bbox_ymax

    crop_reg_xmin = max(0, min(int(crop_reg_xmin), width))
    crop_reg_ymin = max(0, min(int(crop_reg_ymin), height))
    crop_reg_xmax = max(0, min(int(crop_reg_xmax), width))
    crop_reg_ymax = max(0, min(int(crop_reg_ymax), height))

    # If the feasible crop interval becomes invalid, keep the original sample
    # instead of dropping into a debugger or crashing a worker.
    if crop_reg_xmin < 0 or crop_reg_ymin < 0 or crop_reg_xmax > width or crop_reg_ymax > height:
        return img, bbox, gazex, gazey
    if crop_reg_xmin > crop_reg_xmax or crop_reg_ymin > crop_reg_ymax:
        return img, bbox, gazex, gazey

    xmin = random.randint(0, crop_reg_xmin)
    ymin = random.randint(0, crop_reg_ymin)
    xmax = random.randint(crop_reg_xmax, width)
    ymax = random.randint(crop_reg_ymax, height)

    if xmax <= xmin or ymax <= ymin:
        return img, bbox, gazex, gazey

    img = torchvision.transforms.functional.crop(img, ymin, xmin, ymax - ymin, xmax - xmin)
    bbox = [bbox_xmin - xmin, bbox_ymin - ymin, bbox_xmax - xmin, bbox_ymax - ymin]
    gazex = [x - xmin for x in gazex]
    gazey = [y - ymin for y in gazey]

    return img, bbox, gazex, gazey

def horiz_flip(img, bbox, gazex, gazey, inout):
    width, height = img.size
    img = torchvision.transforms.functional.hflip(img)
    xmin, ymin, xmax, ymax = bbox
    bbox = [width - xmax, ymin, width - xmin, ymax]
    if inout:
        gazex = [width - x for x in gazex]
    return img, bbox, gazex, gazey

def random_bbox_jitter(img, bbox):
    width, height = img.size
    xmin, ymin, xmax, ymax = bbox
    jitter = 0.2
    xmin_j = (np.random.random_sample() * (jitter*2) - jitter) * (xmax - xmin)
    xmax_j = (np.random.random_sample() * (jitter*2) - jitter) * (xmax - xmin)
    ymin_j = (np.random.random_sample() * (jitter*2) - jitter) * (ymax - ymin)
    ymax_j = (np.random.random_sample() * (jitter*2) - jitter) * (ymax - ymin)

    bbox = [max(0, xmin_j + xmin), max(0, ymin_j + ymin), min(width, xmax_j + xmax), min(height, ymax_j + ymax)]

    return bbox

def get_heatmap(gazex, gazey, height, width, sigma=3, htype="Gaussian"):
    # Adapted from https://github.com/ejcgt/attention-target-detection/blob/master/utils/imutils.py

    img = torch.zeros(height, width)
    if gazex < 0 or gazey < 0:  # return empty map if out of frame
        return img
    gazex = int(gazex * width)
    gazey = int(gazey * height)

    # Check that any part of the gaussian is in-bounds
    ul = [int(gazex - 3 * sigma), int(gazey - 3 * sigma)]
    br = [int(gazex + 3 * sigma + 1), int(gazey + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if htype == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif htype == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / img.max()  # normalize heatmap so it has max value of 1
    return img

# GazeFollow calculates AUC using original image size with GT (x,y) coordinates set to 1 and everything else as 0
# References:
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_gazefollow.py#L78
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/imutils.py#L67
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/evaluation.py#L7
def gazefollow_auc(heatmap, gt_gazex, gt_gazey, height, width):
    target_map = np.zeros((height, width))
    for point in zip(gt_gazex, gt_gazey):
        if point[0] >= 0:
            x, y = map(int, [point[0]*float(width), point[1]*float(height)])
            x = min(x, width - 1)
            y = min(y, height - 1)
            target_map[y, x] = 1
    resized_heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(dim=0).unsqueeze(dim=0), (height, width), mode='bilinear').squeeze()
    auc = roc_auc_score(target_map.flatten(), resized_heatmap.cpu().flatten())
    
    return auc

# Reference: https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_gazefollow.py#L81
def gazefollow_l2(heatmap, gt_gazex, gt_gazey):
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (heatmap.shape[0], heatmap.shape[1]))
    pred_x = pred_x / float(heatmap.shape[1])
    pred_y = pred_y / float(heatmap.shape[0])

    gazex = np.array(gt_gazex)
    gazey = np.array(gt_gazey)

    avg_l2 = np.sqrt((pred_x - gazex.mean())**2 + (pred_y - gazey.mean())**2)
    all_l2s = np.sqrt((pred_x - gazex)**2 + (pred_y - gazey)**2)
    min_l2 = all_l2s.min().item()

    return avg_l2, min_l2

# VideoAttentionTarget calculates AUC on 64x64 heatmap, defining a rectangular tolerance region of 6*(sigma=3) + 1 (uses 2D Gaussian code but binary thresholds > 0 resulting in rectangle)
# References:
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_videoatttarget.py#L106
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/imutils.py#L31
def vat_auc(heatmap, gt_gazex, gt_gazey):
    res = 64
    sigma = 3
    assert heatmap.shape[0] == res and heatmap.shape[1] == res
    target_map = np.zeros((res, res))
    gazex = gt_gazex * res
    gazey = gt_gazey * res
    ul = [max(0, int(gazex - 3 * sigma)), max(0, int(gazey - 3 * sigma))]
    br = [min(int(gazex + 3 * sigma + 1), res-1), min(int(gazey + 3 * sigma + 1), res-1)]
    target_map[ul[1]:br[1], ul[0]:br[0]] = 1
    auc = roc_auc_score(target_map.flatten(), heatmap.cpu().flatten())
    return auc

# Reference: https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_videoatttarget.py#L118
def vat_l2(heatmap, gt_gazex, gt_gazey):
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (64, 64))
    pred_x = pred_x / 64.
    pred_y = pred_y / 64.

    l2 = np.sqrt((pred_x - gt_gazex)**2 + (pred_y - gt_gazey)**2)

    return l2


def compute_bilinear_weights(x, y, grid_h=32, grid_w=32):
    """Compute bilinear interpolation weights for a point on a grid.

    Args:
        x: normalized x coordinate in [0, 1]
        y: normalized y coordinate in [0, 1]
        grid_h: grid height (default 32 for DINOv2 ViT-B/14 with 448 input)
        grid_w: grid width

    Returns:
        indices: LongTensor [4] - flattened indices of the 4 surrounding grid points
        weights: FloatTensor [4] - bilinear interpolation weights (sum to 1)
    """
    # Map to floating-point grid coordinates (pixel-center aligned)
    col_f = x * grid_w - 0.5
    row_f = y * grid_h - 0.5

    # Clamp to valid range
    col_f = max(0.0, min(col_f, grid_w - 1.0))
    row_f = max(0.0, min(row_f, grid_h - 1.0))

    # Integer coordinates of the top-left anchor
    col_int = int(np.floor(col_f))
    row_int = int(np.floor(row_f))

    # Fractional parts
    dx = col_f - col_int
    dy = row_f - row_int

    # Clamp to ensure we stay within grid bounds
    col_int = min(col_int, grid_w - 2)
    row_int = min(row_int, grid_h - 2)

    # Four surrounding anchors: TL, TR, BL, BR
    tl = row_int * grid_w + col_int
    tr = row_int * grid_w + (col_int + 1)
    bl = (row_int + 1) * grid_w + col_int
    br = (row_int + 1) * grid_w + (col_int + 1)

    # Bilinear interpolation weights (diagonal area rule)
    w_tl = (1 - dx) * (1 - dy)
    w_tr = dx * (1 - dy)
    w_bl = (1 - dx) * dy
    w_br = dx * dy

    indices = torch.tensor([tl, tr, bl, br], dtype=torch.long)
    weights = torch.tensor([w_tl, w_tr, w_bl, w_br], dtype=torch.float32)
    return indices, weights


def compute_geo_loss(anchors_weight, grid_h=32, grid_w=32):
    """Geometric constraint loss on anchor embeddings.

    Encourages the learned anchor embeddings to preserve the spatial geometry
    of the grid: anchors that are physically close should have similar embeddings.

    Args:
        anchors_weight: nn.Embedding weight tensor [grid_h*grid_w, clip_dim]
        grid_h, grid_w: grid dimensions

    Returns:
        loss_geo: scalar L1 loss between pairwise cosine similarity and
                  normalized Euclidean distance matrices
    """
    num_anchors = grid_h * grid_w
    device = anchors_weight.device

    # Build normalized Euclidean distance matrix for the 32x32 grid
    rows = torch.arange(grid_h, device=device, dtype=torch.float32)
    cols = torch.arange(grid_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')  # [H, W]
    grid_y = grid_y.reshape(-1)  # [1024]
    grid_x = grid_x.reshape(-1)  # [1024]

    # Pairwise Euclidean distance
    dy = grid_y.unsqueeze(0) - grid_y.unsqueeze(1)  # [1024, 1024]
    dx = grid_x.unsqueeze(0) - grid_x.unsqueeze(1)
    dist_matrix = torch.sqrt(dx ** 2 + dy ** 2)

    # Normalize to [0, 1]
    max_dist = dist_matrix.max()
    if max_dist > 0:
        dist_matrix = dist_matrix / max_dist

    # Cosine similarity of anchor embeddings
    anchors_norm = torch.nn.functional.normalize(anchors_weight, p=2, dim=1)
    cos_sim = torch.mm(anchors_norm, anchors_norm.t())  # [1024, 1024]

    # L1 loss: cosine similarity should be inversely related to distance
    # Use (1 - dist) as target since close points should have high similarity
    target_sim = 1.0 - dist_matrix
    loss_geo = torch.nn.functional.l1_loss(cos_sim, target_sim)

    return loss_geo


def compute_mcr_loss(vision_tokens, f_text, temperature=0.07):
    """Token-level InfoNCE loss for cross-modal distillation.

    Each spatial vision token must match its own text feature (positive)
    while distinguishing it from text features of other samples (negatives).
    The CLIP text encoder backbone is frozen; only anchors/soft prompts/text_proj
    are learnable and receive gradients through this loss.

    Args:
        vision_tokens: [N, num_tokens, dim] projected vision tokens
        f_text: [N, dim] text feature
        temperature: softmax temperature (lower = sharper)

    Returns:
        loss_mcr: scalar InfoNCE loss
    """
    N, T, D = vision_tokens.shape

    # L2 normalize
    v_norm = torch.nn.functional.normalize(vision_tokens, dim=2)  # [N, T, D]
    t_norm = torch.nn.functional.normalize(f_text, dim=1)          # [N, D]

    # Cosine similarity: each vision token vs each text feature in batch
    # [N, T, D] x [N, D] -> [N, T, N]
    logits = torch.bmm(v_norm, t_norm.unsqueeze(0).expand(N, -1, -1).permute(0, 2, 1)) / temperature
    # logits[b, t, n] = sim(vision_token[b,t], text_feature[n])

    # Positive labels: sample i's tokens match sample i's text
    labels = torch.arange(N, device=vision_tokens.device)  # [N]

    # Flatten tokens: [N*T, N] — each token classifies which sample it belongs to
    logits = logits.reshape(N * T, N)
    labels = labels.unsqueeze(1).expand(-1, T).reshape(N * T)  # [N*T]

    loss_mcr = torch.nn.functional.cross_entropy(logits, labels)
    return loss_mcr
