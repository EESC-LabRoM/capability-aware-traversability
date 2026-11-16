import argparse
from pathlib import Path
import time
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, to_tensor
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
import yaml

from cat.data.depth import prepare_depth_input
from cat.embedding_generation.models import FeatureEmbedder
from cat.training.capability import modulate_semantic_map
from cat.training.checkpoint import build_decoder, load_model_state, strip_compile_prefix

PROFILE_LABELS = {'wheeled': 'Wheeled', 'quadruped': 'Legged', 'ATV': 'ATV', 'differential': 'Differential'}
COLORMAP = plt.get_cmap('RdYlBu')

def load_model(checkpoint_path, device) -> tuple:
    """
    Load CAT and its frozen image encoders.

    :param checkpoint_path: Path - trusted CAT training checkpoint
    :param device: torch.device - inference device
    :return: tuple - decoder, DINO, CLIPSeg, processor, and checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    weights = strip_compile_prefix(checkpoint['model_state_dict'])
    use_depth = any(name.startswith('depth_mixer.') for name in weights)

    decoder = build_decoder(config, len(config['prompts']), use_depth)
    load_model_state(decoder, weights)
    decoder = decoder.to(device).eval()

    backbone = FeatureEmbedder().to(device).eval().dino

    clipseg_name = 'CIDAS/clipseg-rd64-refined'
    processor = CLIPSegProcessor.from_pretrained(clipseg_name)
    clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_name)
    clipseg = clipseg.to(device).eval()

    return decoder, backbone, clipseg, processor, checkpoint

def crop_window(width, height) -> tuple:
    """
    Geometry of the aspect-preserving resize and 224x224 center crop.

    :param width: int - image width
    :param height: int - image height
    :return: tuple - resized (width, height), crop left, crop top
    """
    scale = max(224 / width, 224 / height)
    size = (max(224, int(width * scale)), max(224, int(height * scale)))

    return size, size[0] // 2 - 112, size[1] // 2 - 112

def center_crop(image, interpolation) -> np.ndarray:
    """
    Aspect-preserving resize and center crop to 224x224.

    :param image: np.ndarray - RGB or depth image
    :param interpolation: int - OpenCV resize method
    :return: np.ndarray - cropped image
    """
    size, left, top = crop_window(image.shape[1], image.shape[0])
    image = cv2.resize(image, size, interpolation=interpolation)

    return image[top:top + 224, left:left + 224]

def native_crop_box(width, height) -> tuple:
    """
    The model's 224x224 crop expressed in native image pixels.

    :param width: int - image width
    :param height: int - image height
    :return: tuple - x0, y0, x1, y1
    """
    size, left, top = crop_window(width, height)
    sx, sy = width / size[0], height / size[1]

    return round(left * sx), round(top * sy), round((left + 224) * sx), round((top + 224) * sy)

def prepare_rgb(path, device) -> tuple:
    """
    Prepare RGB inputs for DINO and CLIPSeg.

    :param path: Path - RGB image
    :param device: torch.device - inference device
    :return: tuple - native RGB image, normalized RGB tensor, and CLIPSeg image
    """
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop = center_crop(image, cv2.INTER_AREA)

    rgb = to_tensor(crop)
    rgb = normalize(rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rgb = rgb.unsqueeze(0).to(device)

    # PIL bilinear, as in the CLIPSeg training cache
    semantic_image = Image.fromarray(image)
    scale = max(224 / semantic_image.width, 224 / semantic_image.height)
    size = (max(224, int(semantic_image.width * scale)), max(224, int(semantic_image.height * scale)))
    semantic_image = semantic_image.resize(size, Image.Resampling.BILINEAR)

    left = (size[0] - 224) // 2
    top = (size[1] - 224) // 2
    semantic_image = semantic_image.crop((left, top, left + 224, top + 224))

    return image, rgb, semantic_image

def prepare_depth(path, sensor, stats, device) -> torch.Tensor:
    """
    Build the depth input the checkpoint was trained with from registered depth (mm, 0 = no reading).

    :param path: Path - registered uint16 depth PNG
    :param sensor: str - depth statistics and clipping sensor key
    :param stats: dict - per-sensor depth statistics of the checkpoint's training embeddings
    :param device: torch.device - inference device
    :return: torch.Tensor - [1, 3, H, W] normalized depth repeated over three channels
    """
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    near, far = stats['depth_clip_by_sensor'][sensor]
    mean = stats['depth_mean_by_sensor'][sensor]
    std = stats['depth_std_by_sensor'][sensor]

    if 'depth_min_valid_fraction' in stats:
        depth = prepare_depth_input(depth_mm, (224, 224), (near, far), mean, std, stats['depth_min_valid_fraction'])
        if depth is None:
            print('Depth coverage below depth_min_valid_fraction: using zero depth input')
            depth = np.zeros((224, 224), dtype=np.float32)
    else:
        depth = center_crop(depth_mm, cv2.INTER_NEAREST).astype(np.float32) / 1000
        invalid = depth == 0
        depth = np.clip(depth, near, far)
        depth[invalid] = far
        depth = ((depth - mean) / std).astype(np.float32)

    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    depth = depth.repeat(1, 3, 1, 1)
    return depth.to(device)

def load_scores(path, config, robots, device) -> dict:
    """
    Read robot capability vectors in checkpoint prompt order.

    :param path: Path - per-sequence score YAML
    :param config: dict - checkpoint config with robot name mapping
    :param robots: list[str] - selected robot names
    :param device: torch.device - inference device
    :return: dict - robot names mapped to capability tensors
    """
    with path.open() as stream:
        scores = yaml.safe_load(stream)

    labels = {name: label for label, name in config['robots']['name_map'].items()}
    vectors = {}

    for robot in robots:
        key = robot if robot in scores else labels[robot]
        values = [entry['score'] for entry in scores[key]]
        vectors[robot] = torch.tensor(values, dtype=torch.float32, device=device)

    return vectors

def extract_features(backbone, image) -> torch.Tensor:
    """
    Extract a DINO patch feature grid.

    :param backbone: torch.nn.Module - frozen DINO encoder
    :param image: torch.Tensor - prepared image [1, 3, 224, 224]
    :return: torch.Tensor - features [1, D, 14, 14]
    """
    tokens = backbone.forward_features(image)
    patches = tokens[:, backbone.num_prefix_tokens:]
    features = patches.transpose(1, 2).reshape(1, -1, 14, 14)

    return features.float()

@torch.inference_mode()
def predict(decoder, backbone, clipseg, inputs, rgb, depth, vectors, prototypes) -> torch.Tensor:
    """
    Encode the scene once and predict traversability for each robot.

    :param decoder: torch.nn.Module - trained CAT decoder
    :param backbone: torch.nn.Module - frozen DINO encoder
    :param clipseg: torch.nn.Module - frozen terrain segmentation model
    :param inputs: dict - prepared CLIPSeg inputs
    :param rgb: torch.Tensor - normalized RGB image
    :param depth: torch.Tensor | None - normalized depth image, None for RGB-only
    :param vectors: dict - robot capability tensors
    :param prototypes: dict - learned robot prototypes
    :return: torch.Tensor - cosine maps [R, H, W]
    """
    rgb_features = extract_features(backbone, rgb)

    depth_features = None
    if depth is not None:
        depth_features = extract_features(backbone, depth)

    logits = clipseg(**inputs).logits.float().unsqueeze(0)
    terrain = logits.softmax(dim=1)

    # 8-bit quantize to match the terrain maps stored by training
    terrain = (terrain * 255).byte().float() / 255
    maps = []

    for robot, vector in vectors.items():
        conditioning = modulate_semantic_map(terrain, vector)
        features = decoder(rgb_features, conditioning, depth_features)

        prototype = prototypes[robot].to(rgb.device).float().reshape(-1)
        prototype = F.normalize(prototype, dim=0)
        cosine = torch.einsum('d,bdhw->bhw', prototype, features)[0]

        maps.append(cosine.clamp(-1, 1))

    return torch.stack(maps)

def plot_results(scenes, vectors) -> None:
    """
    Show the RGB inputs and per-robot traversability maps in one figure.

    :param scenes: list[tuple] - per scene: RGB path, native RGB image, cosine maps [R, H, W]
    :param vectors: dict - robot capability tensors in output order
    """
    robots = list(vectors)

    cosine = torch.stack([scene[2] for scene in scenes]).cpu().numpy()
    traversability = (cosine + 1) / 2

    for (path, *_), scores in zip(scenes, traversability):
        print(path)
        for index, robot in enumerate(robots):
            print(f'  {index:02d}: {robot}, mean traversability = {scores[index].mean():.3f}')

    columns = 1 + len(robots)
    figure = plt.figure(figsize=(2.6 * columns, 2.6 * len(scenes) + 0.8), layout='constrained')
    grid = figure.add_gridspec(len(scenes) + 1, columns, height_ratios=[1] * len(scenes) + [0.1])
    titles = ['RGB input'] + [PROFILE_LABELS.get(robot, robot) for robot in robots]

    for row, ((_, image, *_), scores) in enumerate(zip(scenes, traversability)):
        height, width = image.shape[:2]
        x0, y0, x1, y1 = native_crop_box(width, height)
        crop = image[y0:y1, x0:x1].astype(np.float32)

        gray = (crop @ np.float32([0.299, 0.587, 0.114]))[..., None]

        panels = [crop]
        for score in scores:
            heatmap = COLORMAP(score)[..., :3] * 255
            heatmap = cv2.resize(heatmap.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            panels.append(0.5 * gray + 0.5 * heatmap)

        for column, pixels in enumerate(panels):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(pixels.clip(0, 255).astype(np.uint8))
            axis.set_axis_off()
            if row == 0:
                axis.set_title(titles[column], fontsize=11, color='0.15', loc='left')

    figure.canvas.draw()
    figure.set_layout_engine('none')
    first, last = figure.axes[-columns].get_position(), figure.axes[-1].get_position()
    inch = 1 / figure.get_figheight()
    bar = figure.add_axes([first.x0, first.y0 - 0.15 * inch, last.x1 - first.x0, 0.07 * inch])
    bar.imshow(np.linspace(0, 1, 256)[None], cmap=COLORMAP, aspect='auto')
    bar.set_axis_off()
    style = dict(fontsize=9, color='0.4', va='top', transform=bar.transAxes)
    bar.text(0, -0.5, 'low', ha='left', **style)
    bar.text(0.5, -0.5, 'traversability', ha='center', **style)
    bar.text(1, -0.5, 'high', ha='right', **style)

    figure.canvas.manager.set_window_title('CAT demo')
    plt.show()

def main(args) -> None:
    """
    Run CAT inference on one or more images and show them in one figure.

    :param args: argparse.Namespace - demo command-line options
    """
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print(f'Loading CAT on {device}...')
    decoder, backbone, clipseg, processor, checkpoint = load_model(args.checkpoint, device)
    config = checkpoint['config']

    stats = None
    if decoder.use_depth:
        stats_path = Path(args.checkpoint).with_name('depth_stats.json')
        if not stats_path.is_file():
            raise FileNotFoundError(f'Required depth statistics not found: {stats_path}')
        with stats_path.open() as stream:
            stats = json.load(stream)
        if not args.sensor:
            raise ValueError('--sensor is required for RGB-D inference')
        if not args.depth or len(args.depth) != len(args.rgb):
            raise ValueError('RGB-D inference needs one --depth image per --rgb image')

    robots = args.robots or list(checkpoint['z'])
    vectors = load_scores(args.scores, config, robots, device)
    prompts = [prompt['text'] for prompt in config['prompts']]
    scenes = []

    for index, path in enumerate(args.rgb):
        image, rgb, semantic_image = prepare_rgb(path, device)

        depth = None
        if stats is not None:
            depth = prepare_depth(args.depth[index], args.sensor, stats, device)

        inputs = processor(
            text=prompts,
            images=[semantic_image] * len(prompts),
            padding=True,
            return_tensors='pt',
            do_resize=False,
        ).to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        cosine = predict(
            decoder, backbone, clipseg, inputs, rgb, depth, vectors, checkpoint['z'],
        )

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        print(f'Inference {path}: {time.perf_counter() - start:.2f}s')

        scenes.append((path, image, cosine))

    plot_results(scenes, vectors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--rgb', type=Path, nargs='+', required=True, help='One or more RGB images, one figure row each')
    parser.add_argument('--scores', type=Path, required=True, help='Capability YAML in checkpoint prompt order')
    parser.add_argument('--robots', nargs='+', help='Default: all learned robot prototypes')

    parser.add_argument('--depth', type=Path, nargs='+', help='Registered uint16 depth PNGs (mm), one per --rgb image')
    parser.add_argument('--sensor', help='Depth statistics and clipping sensor key')

    parser.add_argument('--device', default='auto')

    main(parser.parse_args())
