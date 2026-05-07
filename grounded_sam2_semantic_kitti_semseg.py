
import argparse
import contextlib
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── 屏蔽无害但嘈杂的警告 ─────────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*indexing.*")
warnings.filterwarnings("ignore", message=".*non-writable tensors.*")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm

try:
    from helpers import get_augmentation
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from helpers import get_augmentation


# ---------------------------------------------------------------------------
# 类别 / Prompt / 阈值定义
# ---------------------------------------------------------------------------
SEMANTIC_PROMPTSET = {
    "Vehicle": [
        "car", "SUV", "van", "bus", "truck", "trailer",
        "engineering vehicle", "construction vehicle", "dump truck",
        "excavator", "crane", "concrete mixer",
    ],
    "Cycle": [
        "bicycle", "motorcycle", "motor scooter", "e-bike",
    ],
    "Pedestrian": [
        "person", "pedestrian", "adult", "child", "rider",
    ],
    "Road": [
        "road", "drivable surface", "lane", "lane marking",
    ],
    "Sidewalk": [
        "sidewalk", "curb", "bike path", "walkway",
        "pavement", "footpath", "footway",
    ],
    "Structure": [
        "building", "building facade", "building exterior", "house", "garage",
        "wall", "concrete wall", "retaining wall", "stairs",
        "railing", "awning", "roof", "bridge",
    ],
    "Vegetation": ["tree", "bush", "shrub", "plant", "flower", "grass"],
    "Traffic Facility": [
        "pole",
        "traffic light pole",
        "street light pole",
        "sign pole",
        "traffic sign",
        "road sign",
        "speed limit sign",
        "traffic light",
        "traffic signal",
    ],
    "Sky": ["sky", "cloudy sky", "overcast sky"],
}

SEMANTIC_CLASS_PROMPTS = [
    {"name": "Vehicle",
     "prompt": SEMANTIC_PROMPTSET["Vehicle"],
     "box_threshold": 0.16, "text_threshold": 0.16},


    {"name": "Cycle",
     "prompt": SEMANTIC_PROMPTSET["Cycle"],
     "box_threshold": 0.22, "text_threshold": 0.22},

    {"name": "Pedestrian",
     "prompt": SEMANTIC_PROMPTSET["Pedestrian"],
     "box_threshold": 0.16, "text_threshold": 0.16},

    {"name": "Road",
     "prompt": SEMANTIC_PROMPTSET["Road"],
     "box_threshold": 0.10, "text_threshold": 0.10},

   
    {"name": "Sidewalk",
     "prompt": SEMANTIC_PROMPTSET["Sidewalk"],
     "box_threshold": 0.13, "text_threshold": 0.13},

    {"name": "Structure",
     "prompt": SEMANTIC_PROMPTSET["Structure"],
     "box_threshold": 0.10, "text_threshold": 0.10},

    {"name": "Vegetation",
     "prompt": SEMANTIC_PROMPTSET["Vegetation"],
     "box_threshold": 0.10, "text_threshold": 0.10},

  
    {"name": "Traffic Facility",
     "prompt": SEMANTIC_PROMPTSET["Traffic Facility"],
     "box_threshold": 0.16, "text_threshold": 0.16},

    {"name": "Sky",
     "prompt": SEMANTIC_PROMPTSET["Sky"],
     "box_threshold": 0.08, "text_threshold": 0.08},
]

SEMANTIC_CLASS_ID_MAP = {
    "Vehicle":          1,
    "Cycle":            2,
    "Pedestrian":       3,
    "Road":             4,
    "Sidewalk":         5,
    "Structure":        6,
    "Vegetation":       8,
    "Traffic Facility": 11,
    "Sky":              13,
}

SEMANTIC_ID_TO_NAME = {v: k for k, v in SEMANTIC_CLASS_ID_MAP.items()}

# 检测框面积过滤：单个 box 占整图面积超过阈值视为误检丢弃
LARGE_BOX_AREA_RATIO = 0.40
# Vegetation / Sky / Road 天然可以占大面积，豁免大框过滤
CLASSES_BYPASS_LARGE_BOX_FILTER = {"Vegetation", "Sky", "Road"}
# Traffic Facility 单独限制更严：真正的杆/灯/标牌不会超过图像面积的 8%
TRAFFIC_FACILITY_MAX_BOX_RATIO = 0.08

IGNORE_ID = 0

# ---------------------------------------------------------------------------
# 跨类 NMS（Cross-class NMS）配置
# ---------------------------------------------------------------------------
# 原理：
#   GDino 对同一块路面既检出了 Road box 又检出了 Sidewalk box，
#   SAM2 随即给两者生成了高度重叠的掩码。
#   merge_global 里纯靠置信度决胜，而置信度本身帧间不稳定，
#   导致同一块区域在不同帧交替被判为 Road 或 Sidewalk（颜色闪烁）。
#
# 解决方案：
#   在 SAM2 出完掩码之后、merge 之前，对"互斥类对"做掩码级 NMS：
#   两个来自不同类的掩码 IoU > 阈值，说明它们描述的是同一块区域，
#   只保留置信度更高的那个，另一个直接丢弃，不参与 merge。
#   这样竞争在掩码级别一次决出胜负，消除帧间抖动。
MUTEX_CLASS_PAIRS = [
    ("Road",    "Sidewalk"),  # 道路 vs 人行道 经常重叠
    ("Vehicle", "Cycle"),     # 车辆 vs 自行车/摩托车 后视角容易混淆
]
CROSS_NMS_IOU_THRESHOLD = 0.40  # IoU 超过此值即视为"同一块区域"

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
OUTPUT_DIR_DEFAULT                = Path("outputs/grounded_sam2_semantic_kitti_semseg")
SAM2_CHECKPOINT_DEFAULT           = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG_DEFAULT         = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG_DEFAULT     = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_DEFAULT = "gdino_checkpoints/groundingdino_swint_ogc.pth"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def normalize_label(label: str) -> str:
    return " ".join(label.lower().strip().rstrip(".").split())


def normalize_prompts(prompts) -> list[str]:
    if isinstance(prompts, str):
        return [prompts]
    return list(prompts)


def get_autocast_context(device: str, use_fp16: bool):
    if device == "cuda" and use_fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# 颜色 / 可视化
# ---------------------------------------------------------------------------
def build_color_map() -> np.ndarray:
    color_map = np.zeros((256, 3), dtype=np.uint8)
    vivid_colors = {
        1:  (220,  20,  60),   # Vehicle           深红
        2:  (255, 140,   0),   # Cycle             橙
        3:  (255,   0, 255),   # Pedestrian        品红
        4:  ( 70, 130, 180),   # Road              钢蓝
        5:  (255, 215,   0),   # Sidewalk          金黄
        6:  (139,  69,  19),   # Structure         棕
        8:  ( 34, 139,  34),   # Vegetation        森林绿
        11: (  0, 255, 255),   # Traffic Facility  青
        13: (135, 206, 235),   # Sky               天蓝
    }
    for class_id, color in vivid_colors.items():
        color_map[class_id] = np.array(color, dtype=np.uint8)
    return color_map


def semantic_mask_to_color(mask: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    return color_map[mask]


def build_overlay(image_rgb: np.ndarray, semantic_mask: np.ndarray,
                  color_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored_mask = semantic_mask_to_color(semantic_mask, color_map)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, colored_mask, alpha, 0.0)


def render_legend_strip(image_width: int, color_map: np.ndarray) -> np.ndarray:
    """生成图例横幅，拼接在可视化图像下方。"""
    items = [(cid, SEMANTIC_ID_TO_NAME[cid]) for cid in sorted(SEMANTIC_ID_TO_NAME.keys())]
    swatch_size, pad_x, pad_y = 18, 8, 8
    row_height = swatch_size + pad_y * 2
    text_pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 0.45, 1

    item_widths = []
    for cid, name in items:
        (tw, _th), _ = cv2.getTextSize(f"{cid}: {name}", font, font_scale, font_thickness)
        item_widths.append(swatch_size + text_pad + tw + pad_x * 2)

    rows, current_row, current_width = [], [], 0
    for idx, w in enumerate(item_widths):
        if current_row and current_width + w > image_width:
            rows.append(current_row); current_row = [idx]; current_width = w
        else:
            current_row.append(idx); current_width += w
    if current_row:
        rows.append(current_row)

    legend_height = row_height * len(rows) + pad_y
    legend = np.full((legend_height, image_width, 3), 245, dtype=np.uint8)
    for row_idx, row in enumerate(rows):
        y0 = pad_y + row_idx * row_height
        x = pad_x
        for item_idx in row:
            cid, name = items[item_idx]
            rgb = color_map[cid]
            bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            cv2.rectangle(legend, (x, y0), (x + swatch_size, y0 + swatch_size), bgr, -1)
            cv2.rectangle(legend, (x, y0), (x + swatch_size, y0 + swatch_size), (60, 60, 60), 1)
            text = f"{cid}: {name}"
            cv2.putText(legend, text, (x + swatch_size + text_pad, y0 + swatch_size - 4),
                        font, font_scale, (20, 20, 20), font_thickness, cv2.LINE_AA)
            (tw, _th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            x += swatch_size + text_pad + tw + pad_x * 2
    return legend


def attach_legend(image_bgr: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    return np.vstack([image_bgr, render_legend_strip(image_bgr.shape[1], color_map)])


# ---------------------------------------------------------------------------
# 数据增强工具
# ---------------------------------------------------------------------------
def _gdino_transform_from_array(image_rgb_np: np.ndarray) -> torch.Tensor:
    """增强后的 RGB numpy array → GroundingDINO 所需的归一化 tensor。"""
    from grounding_dino.groundingdino.datasets import transforms as GDT
    _tf = GDT.Compose([
        GDT.RandomResize([800], max_size=1333),
        GDT.ToTensor(),
        GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil = Image.fromarray(image_rgb_np.astype(np.uint8))
    tensor, _ = _tf(pil, None)
    return tensor


def _flip_mask_back(mask: np.ndarray, aug_mode: str) -> np.ndarray:
    """翻转增强推理完成后，将 mask 翻回原始朝向。"""
    if aug_mode == "horizontal-flip":
        return np.fliplr(mask).copy()
    if aug_mode == "vertical-flip":
        return np.flipud(mask).copy()
    return mask


# ---------------------------------------------------------------------------
# 跨类 NMS
# ---------------------------------------------------------------------------
def cross_class_nms(
    idx_groups: dict[str, list[int]],
    box_meta: list[tuple[str, float, str]],
    all_masks: np.ndarray,
    mutex_pairs: list[tuple[str, str]],
    iou_thr: float,
) -> dict[str, list[int]]:
    """
    对互斥类对执行掩码级别的跨类 NMS。

    遍历每个互斥对 (class_a, class_b)：
      - 对 class_a 的每个掩码 i 和 class_b 的每个掩码 j 计算 IoU
      - 若 IoU > iou_thr，说明两者描述同一区域，保留置信度更高的，丢弃另一个
      - 置信度相同时保留 class_a 的掩码（即列表中第一个类）

    返回过滤后的 idx_groups（被抑制的索引已从列表中移除）。
    """
    suppressed: set[int] = set()

    for class_a, class_b in mutex_pairs:
        indices_a = idx_groups.get(class_a, [])
        indices_b = idx_groups.get(class_b, [])
        if not indices_a or not indices_b:
            continue

        for i in indices_a:
            for j in indices_b:
                if i in suppressed or j in suppressed:
                    continue

                mask_i = all_masks[i].astype(bool)
                mask_j = all_masks[j].astype(bool)

                inter = float((mask_i & mask_j).sum())
                if inter == 0.0:
                    continue

                iou = inter / (float((mask_i | mask_j).sum()) + 1e-6)
                if iou <= iou_thr:
                    continue

                # IoU 超过阈值：置信度低的被抑制，置信度相同则抑制 class_b
                conf_i = box_meta[i][1]
                conf_j = box_meta[j][1]
                suppressed.add(j if conf_i >= conf_j else i)

    # 从各类的索引列表中移除被抑制的掩码
    return {
        cls: [idx for idx in idxs if idx not in suppressed]
        for cls, idxs in idx_groups.items()
    }


# ---------------------------------------------------------------------------
# 后处理
# ---------------------------------------------------------------------------
POSTPROCESS_VEG_FILL_KERNEL = 7
POSTPROCESS_DOMINANT_CLASSES = ("Sky", "Road", "Vegetation")
POSTPROCESS_DOMINANT_MIN_SECONDARY_AREA = 1500


def fill_holes_in_class(semantic_mask, prompt_id_map, target_class, kernel_size=7):
    """对指定类别的掩码做形态学闭运算，填补内部空洞。"""
    binary = (semantic_mask == target_class).astype(np.uint8)
    if binary.sum() == 0:
        return semantic_mask, prompt_id_map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    new_pixels = (closed == 1) & (binary == 0)
    if new_pixels.any():
        semantic_mask[new_pixels] = target_class
        prompt_id_map[new_pixels] = 0
    return semantic_mask, prompt_id_map


def enforce_dominant_blobs(semantic_mask, prompt_id_map, target_class,
                            min_secondary_area=1500):
    """
    对指定类别保留最大连通域，面积小于 min_secondary_area 的零散小区域
    视为噪声，重置为 IGNORE_ID。
    """
    binary = (semantic_mask == target_class).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return semantic_mask, prompt_id_map
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.argmax()) + 1
    for i in range(1, n):
        if i == largest:
            continue
        if stats[i, cv2.CC_STAT_AREA] < min_secondary_area:
            blob = labels == i
            semantic_mask[blob] = IGNORE_ID
            prompt_id_map[blob] = 0
    return semantic_mask, prompt_id_map


def fix_sidewalk_road_confusion(
    semantic_mask: np.ndarray,
    prompt_id_map: np.ndarray,
    sidewalk_road_ratio_thr: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    驾驶场景语义修正：Sidewalk 面积不可能比 Road 大好几倍。

    如果 Sidewalk 像素数 > Road 像素数 × sidewalk_road_ratio_thr，
    说明大片路面被误判为 Sidewalk（GDino 把宽阔路面检出为 pavement/footway）。
    此时把最大的 Sidewalk 连通域强制改回 Road，
    保留其余小连通域（它们很可能是真实的人行道）。

    sidewalk_road_ratio_thr：
        默认 3.0，即 Sidewalk 面积超过 Road 的 3 倍才触发。
        调低（如 2.0）更激进，调高（如 5.0）更保守。
    """
    road_id     = SEMANTIC_CLASS_ID_MAP["Road"]
    sidewalk_id = SEMANTIC_CLASS_ID_MAP["Sidewalk"]

    n_road     = int((semantic_mask == road_id).sum())
    n_sidewalk = int((semantic_mask == sidewalk_id).sum())

    if n_sidewalk == 0:
        return semantic_mask, prompt_id_map
    if n_road == 0 or n_sidewalk <= n_road * sidewalk_road_ratio_thr:
        return semantic_mask, prompt_id_map

    # 找出最大 Sidewalk 连通域，改为 Road
    binary = (semantic_mask == sidewalk_id).astype(np.uint8)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_comp <= 1:
        return semantic_mask, prompt_id_map

    largest_comp = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
    biggest_blob = labels == largest_comp
    semantic_mask[biggest_blob] = road_id
    prompt_id_map[biggest_blob] = 0
    return semantic_mask, prompt_id_map


def postprocess_semantic_mask(semantic_mask, prompt_id_map):
    """后处理：Vegetation 填孔 + Sky/Road/Vegetation 主连通域清理 + Road/Sidewalk 混淆修正。"""
    semantic_mask, prompt_id_map = fill_holes_in_class(
        semantic_mask, prompt_id_map,
        target_class=SEMANTIC_CLASS_ID_MAP["Vegetation"],
        kernel_size=POSTPROCESS_VEG_FILL_KERNEL,
    )
    for cls_name in POSTPROCESS_DOMINANT_CLASSES:
        semantic_mask, prompt_id_map = enforce_dominant_blobs(
            semantic_mask, prompt_id_map,
            target_class=SEMANTIC_CLASS_ID_MAP[cls_name],
            min_secondary_area=POSTPROCESS_DOMINANT_MIN_SECONDARY_AREA,
        )
    # Sidewalk 面积远超 Road 时，最大 Sidewalk 连通域改回 Road
    semantic_mask, prompt_id_map = fix_sidewalk_road_confusion(
        semantic_mask, prompt_id_map,
        sidewalk_road_ratio_thr=3.0,
    )
    return semantic_mask, prompt_id_map


# ---------------------------------------------------------------------------
# 置信度竞争合并（纯净版，无任何 hack）
# ---------------------------------------------------------------------------
def merge_global(semantic_mask, score_map, prompt_id_map,
                 instance_masks, confidences, class_id, prompt_ids):
    """
    同一像素被多个掩码覆盖时，置信度更高的掩码获胜。
    跨类竞争已由 cross_class_nms 在合并前解决，
    此处只处理同类内部多个实例之间的竞争。
    """
    order = np.argsort(np.asarray(confidences))[::-1]
    for idx in order:
        current_mask = instance_masks[idx].astype(bool)
        current_score = float(confidences[idx])
        update_mask = current_mask & (current_score >= score_map)
        semantic_mask[update_mask] = class_id
        score_map[update_mask] = current_score
        prompt_id_map[update_mask] = prompt_ids[idx]
    return semantic_mask, score_map, prompt_id_map


# ---------------------------------------------------------------------------
# 核心推理流水线
# ---------------------------------------------------------------------------
def process_single_image(
    img_path: str,
    grounding_model,
    sam2_predictor,
    device: str,
    use_fp16: bool,
    preloaded_image=None,
    collect_debug: bool = False,
    augmentation=None,
    aug_mode: str | None = None,
    sam2_batch_limit: int = 256,
    cross_nms_iou: float = CROSS_NMS_IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """
    单张图像推理流水线：

    Phase 1 — GDino（9次调用）
        每个类的所有 alias 打包成一个 ". " 分隔的 caption，一次 GDino 调用。
        GDino 图像骨干特征在第一次调用时缓存，后续 8 次直接复用，
        避免重复做最耗时的图像编码（~68次 → 9次）。

    Phase 2 — SAM2（1次批量调用）
        把所有类的检测框拼成一个大数组，一次 predict() 搞定。
        GPU 利用率大幅提升。

    Phase 3 — Cross-class NMS
        对 MUTEX_CLASS_PAIRS 中的互斥类对，检查掩码两两 IoU：
        IoU > CROSS_NMS_IOU_THRESHOLD → 判定为同一区域，
        删掉置信度低的那个，不再参与 merge。
        从根本上消除 Road/Sidewalk、Vehicle/Cycle 的帧间抖动。

    Phase 4 — Merge + 后处理
        merge_global 做同类内部多实例置信度竞争；
        postprocess_semantic_mask 做形态学后处理（填孔、去噪）。
    """
    # ── 加载 / 增强图像 ──────────────────────────────────────────────────────
    if preloaded_image is None:
        image_source, image = load_image(img_path)
    else:
        image_source, image = preloaded_image

    if augmentation is not None:
        image_source = augmentation(image=image_source)["image"]
        image = _gdino_transform_from_array(image_source)

    sam2_predictor.set_image(image_source)

    height, width = image_source.shape[:2]
    image_area = float(height * width)
    n_classes = len(SEMANTIC_CLASS_PROMPTS)

    # ── Phase 1: GDino 9次调用，图像特征缓存复用 ────────────────────────────
    # 结果存为 class_name → (boxes_np [M,4], confidences_np [M], labels [M])
    class_detections: dict[str, tuple] = {}

    for cls_idx, entry in enumerate(SEMANTIC_CLASS_PROMPTS):
        class_name = entry["name"]
        aliases = normalize_prompts(entry["prompt"])

        # 所有 alias 打包成 GDino 原生支持的 ". " 分隔格式
        caption = " . ".join(aliases) + " ."

        # 只在最后一个类调用后清除图像特征缓存，前 8 次保留缓存供复用
        is_last_class = (cls_idx == n_classes - 1)

        with torch.inference_mode(), get_autocast_context(device, use_fp16):
            boxes, confidences, labels = predict(
                model=grounding_model,
                image=image,
                caption=caption,
                box_threshold=entry["box_threshold"],
                text_threshold=entry["text_threshold"],
                device=device,
                unset_image_tensor=is_last_class,
            )

        if len(boxes) == 0:
            continue

        boxes_px = boxes * torch.tensor(
            [width, height, width, height], device=boxes.device
        )
        input_boxes = box_convert(boxes_px, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        confidences_np = confidences.cpu().numpy()

        # 面积过滤：过大的框视为误检丢弃
        if class_name == "Traffic Facility":
            # Traffic Facility 用更严格的面积上限
            thr = TRAFFIC_FACILITY_MAX_BOX_RATIO * image_area
        elif class_name in CLASSES_BYPASS_LARGE_BOX_FILTER:
            # Vegetation / Sky / Road 不做面积过滤
            thr = None
        else:
            thr = LARGE_BOX_AREA_RATIO * image_area

        if thr is not None:
            box_areas = (
                (input_boxes[:, 2] - input_boxes[:, 0]) *
                (input_boxes[:, 3] - input_boxes[:, 1])
            )
            keep = box_areas <= thr
            if not keep.any():
                continue
            input_boxes    = input_boxes[keep]
            confidences_np = confidences_np[keep]
            labels         = [lbl for lbl, k in zip(labels, keep.tolist()) if k]

        class_detections[class_name] = (input_boxes, confidences_np, labels)

    # 无任何检测结果，提前返回空 mask
    if not class_detections:
        return (
            np.zeros((height, width), dtype=np.uint8),
            image_source,
            {"prompt_id_map": np.zeros((height, width), dtype=np.int32),
             "prompt_registry": [("", "")]} if collect_debug else None,
        )

    # ── Phase 2: SAM2 单次批量推理 ──────────────────────────────────────────
    # 把所有类的 box 拼到一起，记录每个 box 的元信息
    all_boxes: list[np.ndarray] = []
    box_meta: list[tuple[str, float, str]] = []  # (类名, 置信度, 匹配到的 label)

    for class_name, (boxes_np, confs_np, lbls) in class_detections.items():
        for i in range(len(boxes_np)):
            all_boxes.append(boxes_np[i])
            box_meta.append((class_name, float(confs_np[i]), lbls[i]))

    all_boxes_np = np.stack(all_boxes)  # (N_total, 4)

    # 分块调用以防 VRAM OOM（默认每批最多 256 个 box）
    all_masks_list: list[np.ndarray] = []
    for start in range(0, len(all_boxes_np), sam2_batch_limit):
        sub_boxes = all_boxes_np[start: start + sam2_batch_limit]
        with torch.inference_mode(), get_autocast_context(device, use_fp16):
            masks_chunk, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=sub_boxes,
                multimask_output=False,
            )
        if masks_chunk.ndim == 4:
            masks_chunk = masks_chunk.squeeze(1)   # (k, H, W)
        all_masks_list.append(masks_chunk)

    all_masks = np.concatenate(all_masks_list, axis=0)  # (N_total, H, W)

    # ── Phase 3: Cross-class NMS ─────────────────────────────────────────────
    # 按类分组索引
    idx_groups: dict[str, list[int]] = defaultdict(list)
    for global_idx, (class_name, _conf, _lbl) in enumerate(box_meta):
        idx_groups[class_name].append(global_idx)

    # 互斥类对 NMS：IoU > 阈值则删掉置信度低的掩码
    idx_groups = cross_class_nms(
        idx_groups=idx_groups,
        box_meta=box_meta,
        all_masks=all_masks,
        mutex_pairs=MUTEX_CLASS_PAIRS,
        iou_thr=cross_nms_iou,
    )

    # ── Phase 4: Merge + 后处理 ──────────────────────────────────────────────
    semantic_mask = np.zeros((height, width), dtype=np.uint8)
    score_map     = np.zeros((height, width), dtype=np.float32)
    prompt_id_map = np.zeros((height, width), dtype=np.int32)
    prompt_registry: list[tuple[str, str]] = [("", "")]  # 索引 0 占位

    for entry in SEMANTIC_CLASS_PROMPTS:
        class_name = entry["name"]
        indices = idx_groups.get(class_name, [])
        if not indices:
            continue
        class_id = SEMANTIC_CLASS_ID_MAP[class_name]

        instance_masks  = all_masks[indices]
        confidences     = [box_meta[i][1] for i in indices]
        winning_prompts = [box_meta[i][2] for i in indices]

        prompt_ids: list[int] = []
        for pt in winning_prompts:
            prompt_registry.append((class_name, pt))
            prompt_ids.append(len(prompt_registry) - 1)

        semantic_mask, score_map, prompt_id_map = merge_global(
            semantic_mask, score_map, prompt_id_map,
            instance_masks, confidences, class_id, prompt_ids,
        )

    semantic_mask, prompt_id_map = postprocess_semantic_mask(semantic_mask, prompt_id_map)

    # 翻转增强：推理完把 mask 翻回原始朝向
    if aug_mode in ("horizontal-flip", "vertical-flip"):
        semantic_mask = _flip_mask_back(semantic_mask, aug_mode)

    debug_info = None
    if collect_debug:
        debug_info = {"prompt_id_map": prompt_id_map, "prompt_registry": prompt_registry}

    return semantic_mask, image_source, debug_info


# ---------------------------------------------------------------------------
# Debug 可视化（在 overlay 上写出每个区域匹配到的 prompt 文字）
# ---------------------------------------------------------------------------
def render_debug_overlay(image_source, semantic_mask, color_map,
                          prompt_id_map, prompt_registry,
                          min_region_area: int = 800) -> np.ndarray:
    overlay = build_overlay(image_source, semantic_mask, color_map)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    for pid in np.unique(prompt_id_map):
        if pid == 0:
            continue
        binary = (prompt_id_map == pid).astype(np.uint8)
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        class_name, prompt_text = prompt_registry[pid]
        text = prompt_text if prompt_text else class_name

        for comp_idx in range(1, num_labels):
            if stats[comp_idx, cv2.CC_STAT_AREA] < min_region_area:
                continue
            cx, cy = centroids[comp_idx]
            cv2.putText(overlay_bgr, text, (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay_bgr, text, (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay_bgr


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        "Grounded SAM 2 SemanticKITTI semseg (9-class) v4", add_help=True
    )
    parser.add_argument("--input_dir",  type=str, required=True,
                        help="SemanticKITTI 根目录，如 .../dataset/sequences")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--views", type=str, nargs="*",
                        default=["00","01","02","03","04","05","06","07","08","09","10"])
    parser.add_argument("--frame_ids", type=int, nargs="*", default=None,
                        help="只处理指定帧号，如 --frame_ids 0 1 2")
    parser.add_argument("--aug_type",  type=str, default="None",
                        help="增强类型，如 horizontal-flip / clahe，默认不增强")
    parser.add_argument("--sam2_checkpoint",           type=str, default=SAM2_CHECKPOINT_DEFAULT)
    parser.add_argument("--sam2_config",               type=str, default=SAM2_MODEL_CONFIG_DEFAULT)
    parser.add_argument("--grounding_dino_config",     type=str, default=GROUNDING_DINO_CONFIG_DEFAULT)
    parser.add_argument("--grounding_dino_checkpoint", type=str, default=GROUNDING_DINO_CHECKPOINT_DEFAULT)
    parser.add_argument("--use_fp16",  action=argparse.BooleanOptionalAction, default=True,
                        help="开启 CUDA FP16 推理（默认开启）")
    parser.add_argument("--compile",   action="store_true",
                        help="对两个模型应用 torch.compile（需 PyTorch >= 2.0，"
                             "前几张图有 JIT 预热耗时，大数据集值得开启）")
    parser.add_argument("--sam2_batch_limit", type=int, default=256,
                        help="SAM2 单次批量最多处理的 box 数，VRAM 不足时调低（默认 256）")
    parser.add_argument("--cross_nms_iou", type=float, default=CROSS_NMS_IOU_THRESHOLD,
                        help=f"跨类 NMS IoU 阈值，越小越激进（默认 {CROSS_NMS_IOU_THRESHOLD}）")
    parser.add_argument("--prefetch_workers",     type=int, default=2,
                        help="图像预加载 CPU 线程数（默认 2）")
    parser.add_argument("--save_color_preview_n", type=int, default=0,
                        help="保存前 N 张彩色预览（color + overlay），0 表示不保存")
    parser.add_argument("--save_debug_preview_n", type=int, default=0,
                        help="保存前 N 张 debug overlay（写出 prompt 文字），0 表示不保存")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已有输出，默认跳过（断点续跑模式）")
    args = parser.parse_args()

    # ── 增强解析 ─────────────────────────────────────────────────────────────
    aug_type_str = str(args.aug_type).strip()
    if aug_type_str in ("", "None", "none"):
        augmentation, aug_mode = None, None
    else:
        augmentation = get_augmentation(aug_type_str)
        aug_mode = aug_type_str
        print(f"[aug] mode = {aug_mode}")

    # flip 增强时禁用预加载线程（flip 后需重走 GDino 预处理，预加载意义不大）
    prefetch_workers = (
        0 if aug_mode in ("horizontal-flip", "vertical-flip") else args.prefetch_workers
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── CUDA 设置 ─────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    grounding_model = load_model(
        model_config_path=args.grounding_dino_config,
        model_checkpoint_path=args.grounding_dino_checkpoint,
        device=device,
    )

    if args.compile and hasattr(torch, "compile"):
        print("[compile] 正在对 GDino 和 SAM2 应用 torch.compile "
              "（前几张图有 JIT 预热，之后会加速约 20-30%）...")
        grounding_model = torch.compile(grounding_model, mode="reduce-overhead")
        sam2_model      = torch.compile(sam2_model,      mode="reduce-overhead")

    frame_id_filter = set(args.frame_ids) if args.frame_ids else None

    # ── 图像迭代器（含 CPU 预加载）────────────────────────────────────────────
    def iter_images(image_dir: str, view_output_dir: Path):
        image_names = [
            n for n in sorted(os.listdir(image_dir))
            if os.path.isfile(os.path.join(image_dir, n))
        ]
        if frame_id_filter is not None:
            image_names = [
                n for n in image_names
                if Path(n).stem.isdigit() and int(Path(n).stem) in frame_id_filter
            ]
        if not args.overwrite:
            kept, skipped = [], 0
            for n in image_names:
                if (view_output_dir / f"{Path(n).stem}.png").exists():
                    skipped += 1
                else:
                    kept.append(n)
            if skipped:
                print(f"  断点续跑：跳过 {skipped} 个已有 mask（{view_output_dir}）")
            image_names = kept
        if not image_names:
            return
        if prefetch_workers <= 0:
            for n in image_names:
                yield n, None
            return
        max_inflight = max(1, prefetch_workers * 2)
        with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
            inflight = []
            it = iter(image_names)
            for _ in range(max_inflight):
                try:
                    n = next(it)
                except StopIteration:
                    break
                inflight.append((n, executor.submit(load_image, os.path.join(image_dir, n))))
            while inflight:
                n, future = inflight.pop(0)
                yield n, future.result()
                try:
                    nn = next(it)
                    inflight.append((nn, executor.submit(load_image, os.path.join(image_dir, nn))))
                except StopIteration:
                    pass

    color_map = build_color_map()
    global_image_counter = 0

    for view in tqdm(args.views, desc="sequences", unit="seq"):
        view_path = os.path.join(args.input_dir, view, "image_2")
        if not os.path.exists(view_path):
            print(f"跳过不存在的路径：{view_path}")
            continue

        view_output_dir = output_root / view
        view_output_dir.mkdir(parents=True, exist_ok=True)

        image_iter = list(iter_images(view_path, view_output_dir))
        for image_name, image_data in tqdm(
            image_iter, desc=f"seq {view}", unit="img", leave=False
        ):
            image_path = os.path.join(view_path, image_name)
            need_color = global_image_counter < args.save_color_preview_n
            need_debug = global_image_counter < args.save_debug_preview_n

            semantic_mask, image_source, debug_info = process_single_image(
                img_path=image_path,
                grounding_model=grounding_model,
                sam2_predictor=sam2_predictor,
                device=device,
                use_fp16=args.use_fp16,
                preloaded_image=image_data if isinstance(image_data, tuple) else None,
                collect_debug=need_debug,
                augmentation=augmentation,
                aug_mode=aug_mode,
                sam2_batch_limit=args.sam2_batch_limit,
                cross_nms_iou=args.cross_nms_iou,
            )

            stem = Path(image_name).stem
            # 保存语义 mask（灰度图，像素值 = class ID）
            cv2.imwrite(str(view_output_dir / f"{stem}.png"), semantic_mask)

            if need_color:
                color_img   = semantic_mask_to_color(semantic_mask, color_map)
                color_bgr   = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                overlay_img = build_overlay(image_source, semantic_mask, color_map)
                overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(view_output_dir / f"{stem}_color.png"),
                            attach_legend(color_bgr, color_map))
                cv2.imwrite(str(view_output_dir / f"{stem}_overlay.png"),
                            attach_legend(overlay_bgr, color_map))

            if need_debug and debug_info is not None:
                debug_img = render_debug_overlay(
                    image_source=image_source,
                    semantic_mask=semantic_mask,
                    color_map=color_map,
                    prompt_id_map=debug_info["prompt_id_map"],
                    prompt_registry=debug_info["prompt_registry"],
                )
                cv2.imwrite(str(view_output_dir / f"{stem}_debug.png"),
                            attach_legend(debug_img, color_map))

            global_image_counter += 1


if __name__ == "__main__":
    main()