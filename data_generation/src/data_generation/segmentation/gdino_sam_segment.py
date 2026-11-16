import argparse
import copy
import gc
import json
import os
import shutil

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from data_generation.configuration import load_config
from data_generation.utils.common_utils import CommonUtils, update_positive_mask_references
from data_generation.utils.ego_mask import prepare_ego_mask, render_review_panel, review_overlay
from data_generation.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

def check_hil_necessity(current_mask, previous_mask, scores, config_proc) -> tuple[bool, str]:
    """
    Decide whether a frame needs human-in-the-loop review.

    :param current_mask: np.ndarray - (H, W) binary mask, current step
    :param previous_mask: np.ndarray | None - (H, W) binary mask, previous step
    :param scores: np.ndarray - detection scores
    :param config_proc: dict - needs hil_score_threshold, hil_iou_threshold, hil_diff_threshold
    :return: tuple[bool, str] - review flag and reason (empty if none)
    """
    if scores.mean() < config_proc["hil_score_threshold"]:
        return True, f"Low confidence ({scores.mean():.2f})"

    if previous_mask is not None:
        curr = np.array(current_mask)
        prev = np.array(previous_mask)
        intersection = np.logical_and(curr, prev).sum()
        union = np.logical_or(curr, prev).sum()
        iou = intersection / (union + 1e-6)

        if iou < config_proc["hil_iou_threshold"]:
            return True, f"Temporal inconsistency (IoU: {iou:.2f})"

        diff_count = np.count_nonzero(np.logical_xor(curr, prev))
        diff_ratio = diff_count / curr.size

        if diff_ratio > config_proc["hil_diff_threshold"]:
            return True, f"Sudden change (Diff: {diff_ratio:.2%})"

    return False, ""

def process_dataset(dataset_dir, config, device, video_predictor, image_predictor, processor, grounding_model, text, positive_prompts, resume) -> None:
    """
    Label one dataset with Grounding DINO, SAM 2 and video propagation.

    :param dataset_dir: str - processed dataset directory with rgb/
    :param config: dict - generation config with CLI overrides
    :param device: str - torch device
    :param video_predictor: SAM2VideoPredictor - propagates masks across frames
    :param image_predictor: SAM2ImagePredictor - box/point to mask
    :param processor: AutoProcessor - Grounding DINO input processor
    :param grounding_model: AutoModelForZeroShotObjectDetection - Grounding DINO detector
    :param text: str - detection prompt
    :param positive_prompts: list[str] - lowercase traversable class names
    :param resume: str | None - "true" continues from the first frame without a mask; outputs are otherwise cleared unless a start_frame/max_images range is set
    """
    bag_name = os.path.basename(dataset_dir).replace("dataset_", "")
    positive_mask_prefix = config["data_format"]["positive_mask_prefix"]

    video_dir = dataset_dir
    output_dir = dataset_dir

    if not os.path.exists(video_dir):
        return

    os.makedirs(output_dir, exist_ok=True)
    positive_mask_dir = os.path.join(output_dir, "positive_masks")
    json_data_dir = os.path.join(output_dir, "json_data")
    os.makedirs(positive_mask_dir, exist_ok=True)
    os.makedirs(json_data_dir, exist_ok=True)
    rgb_dir = os.path.join(video_dir, "rgb")
    if not os.path.exists(rgb_dir):
         print(f"RGB directory not found in {video_dir}. Skipping.")
         return

    frame_names = [
        p for p in os.listdir(rgb_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"] and p.startswith("rgb_")
    ]

    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split("_")[-1]))
    except ValueError:
        frame_names.sort()

    if not frame_names:
        print(f"No RGB frames found in {rgb_dir}. Skipping.")
        return
    with Image.open(os.path.join(rgb_dir, frame_names[0])) as first_image:
        ego = prepare_ego_mask(dataset_dir, config, first_image.size[::-1])
    print(f"{bag_name}: {ego.description}; excluded pixels are ignored, not non-traversable.")
    ego_valid_tensor = None

    start_frame = config["processing"].get("start_frame", 0)

    max_images = config["processing"].get("max_images")
    partial_range = start_frame > 0 or max_images is not None

    sam_marker = os.path.join(output_dir, "sam_completed.txt")
    if os.path.exists(sam_marker):
        os.remove(sam_marker)
    if (resume is None or resume.lower() != "true") and not partial_range:
        for directory in (positive_mask_dir, json_data_dir):
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
    if start_frame > 0:
        filtered_frames = []
        for p in frame_names:
            try:
                 idx = int(os.path.splitext(p)[0].split("_")[-1])
                 if idx >= start_frame:
                     filtered_frames.append(p)
            except ValueError:
                 filtered_frames.append(p)
        frame_names = filtered_frames

    if resume is not None and resume.lower() == "true":
        for first_missing, name in enumerate(frame_names):
            mask_name = f"{positive_mask_prefix}{os.path.splitext(name)[0]}.tif"
            if not os.path.isfile(os.path.join(positive_mask_dir, mask_name)):
                frame_names = frame_names[first_missing:]
                break
        else:
            frame_names = []

    if max_images is not None:
        frame_names = frame_names[:max_images]

    if config["prompts"].get("mode") != "positive":
        raise ValueError(f"Unsupported prompts.mode: {config['prompts'].get('mode')!r}; expected 'positive'")
    step = config["grounding_dino"].get("step_positive", 5)
    batch_size = int(config["processing"].get("batch_size", 100))
    if batch_size < step:
        batch_size = step
    temp_batch_dir = os.path.join(output_dir, "temp_batch")
    previous_step_mask = None

    for batch_start_idx in range(0, len(frame_names), batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, len(frame_names))
        current_batch_frames = frame_names[batch_start_idx : batch_end_idx]

        if os.path.exists(temp_batch_dir):
            shutil.rmtree(temp_batch_dir)
        os.makedirs(temp_batch_dir, exist_ok=True)

        temp_to_original_mapping = {}
        try:
            for idx, fname in enumerate(current_batch_frames):
                src_path = os.path.abspath(os.path.join(video_dir, "rgb", fname))
                temp_fname = f"{idx:05d}{os.path.splitext(fname)[1]}"
                dst_path = os.path.join(temp_batch_dir, temp_fname)
                os.symlink(src_path, dst_path)
                temp_to_original_mapping[idx] = fname
        except OSError as e:
            raise RuntimeError(f"Failed to create temporary batch for {current_batch_frames}: {e}") from e

        try:
            inference_state = video_predictor.init_state(
                video_path=temp_batch_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
                async_loading_frames=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM video batch {current_batch_frames}: {e}") from e

        sam2_masks = MaskDictionaryModel()
        objects_count = 0

        for local_start_idx in range(0, len(current_batch_frames), step):
            first_frame_in_step = temp_to_original_mapping[local_start_idx]
            image_base_name = os.path.splitext(first_frame_in_step)[0]
            mask_dict = MaskDictionaryModel(promote_type = config["processing"]["video_prompt_type"], mask_name = f"mask_{image_base_name}.npy")

            img_path = os.path.join(video_dir, "rgb", first_frame_in_step)
            image = Image.open(img_path)

            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=config["grounding_dino"]["box_threshold"],
                text_threshold=config["grounding_dino"]["text_threshold"],
                target_sizes=[image.size[::-1]]
            )

            image_predictor.set_image(np.array(image.convert("RGB")))
            input_boxes = results[0]["boxes"]
            OBJECTS = results[0]["text_labels"]


            points_batch = None
            labels_batch = None

            use_traj = config["processing"].get("use_trajectory_points", False)
            is_positive_mode = config["prompts"].get("mode") in ["positive"]

            if use_traj and is_positive_mode and input_boxes.shape[0] > 0:
                traj_mask_dir = video_dir
                traj_name = image_base_name.replace("rgb_", "trajectory_") + ".jpg"
                traj_path = os.path.join(traj_mask_dir, "trajectory_masks", traj_name)

                if not os.path.exists(traj_path):
                     traj_path = os.path.join(traj_mask_dir, traj_name)

                if os.path.exists(traj_path):
                    traj_mask = cv2.imread(traj_path, cv2.IMREAD_GRAYSCALE)
                    if traj_mask is not None:
                        if traj_mask.shape != image.size[::-1]:
                            traj_mask = cv2.resize(traj_mask, image.size, interpolation=cv2.INTER_NEAREST)
                        traj_mask = ego.exclude((traj_mask > 127).astype(np.uint8))
                        num_points_to_sample = config["processing"].get("num_trajectory_points", 5)

                        sampled_points_list = []
                        sampled_labels_list = []


                        for i in range(input_boxes.shape[0]):
                            box = input_boxes[i]
                            x1, y1, x2, y2 = box.int().tolist()
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(image.size[0], x2); y2 = min(image.size[1], y2)

                            crop = traj_mask[y1:y2, x1:x2]
                            ys, xs = np.where(crop > 0)

                            current_points = []
                            current_labels = []
                            if len(xs) > 0:
                                indices = np.linspace(0, len(xs) - 1, num_points_to_sample, dtype=int)
                                for idx in indices:
                                    gx = x1 + xs[idx]
                                    gy = y1 + ys[idx]
                                    current_points.append([gx, gy])
                                    current_labels.append(1)

                            num_found = len(current_points)
                            if num_found < num_points_to_sample:
                                    for _ in range(num_points_to_sample - num_found):
                                        current_points.append([0, 0])
                                        current_labels.append(-1)

                            sampled_points_list.append(current_points)
                            sampled_labels_list.append(current_labels)

                        points_batch = np.array(sampled_points_list)
                        labels_batch = np.array(sampled_labels_list)

            masks = np.zeros((0, image.size[1], image.size[0]), dtype=bool)
            scores = np.zeros((0,), dtype=np.float32)
            logits = None
            if input_boxes.shape[0] != 0 or config["processing"].get("human_in_the_loop", False):
                if input_boxes.shape[0] != 0:
                    masks, scores, logits = image_predictor.predict(
                        point_coords=points_batch,
                        point_labels=labels_batch,
                        box=input_boxes,
                        multimask_output=False,
                    )
                    if masks.ndim == 2:
                        masks = masks[None]
                        scores = scores[None]
                        logits = logits[None]
                    elif masks.ndim == 4:
                        masks = masks.squeeze(1)
                    masks = ego.exclude(masks)

                if config["processing"].get("human_in_the_loop", False):
                    if masks.shape[0] == 0:
                        should_run_hil = True
                        reason = "No detected objects"
                    else:
                        current_mask_binary = np.any(masks > 0, axis=0).astype(np.uint8)
                        should_run_hil, reason = check_hil_necessity(current_mask_binary, previous_step_mask, scores, config["processing"])

                    if batch_start_idx == 0 and local_start_idx == 0:
                        should_run_hil = True
                        reason = "First frame of the sequence (Forced HIL)"

                    if should_run_hil:
                        print(f"--- [HIL REQUIRED] Reason: {reason} ---")
                        current_user_pts = []
                        current_user_lbls = []
                        current_boxes = []

                        state = {
                            'interaction_mode': 'point',
                            'drawing': False,
                            'ix': -1,
                            'iy': -1,
                            'current_box_start': None,
                            'brush_radius': 15
                        }

                        manual_mask_add = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
                        manual_mask_remove = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

                        def on_mouse(event, x, y, flags, param) -> None:
                            """
                            Handle review-window mouse events.

                            :param event: int - OpenCV event code
                            :param x: int - cursor column
                            :param y: int - cursor row
                            :param flags: int - OpenCV event flags
                            :param param: Any - unused
                            """
                            if x < 0 or x >= image_width or y < 0 or y >= image_height:
                                if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                                    state['drawing'] = False
                                    state['current_box_start'] = None
                                return
                            state['ix'], state['iy'] = x, y
                            if (ego.mask is not None and ego.mask[y, x]
                                    and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN)):
                                return
                            if state['interaction_mode'] == 'point':
                                if event == cv2.EVENT_LBUTTONDOWN:
                                    current_user_pts.append([x, y])
                                    current_user_lbls.append(1)
                                elif event == cv2.EVENT_RBUTTONDOWN:
                                    current_user_pts.append([x, y])
                                    current_user_lbls.append(0)
                            elif state['interaction_mode'] == 'box':
                                if event == cv2.EVENT_LBUTTONDOWN:
                                    state['drawing'] = True
                                    state['current_box_start'] = (x, y)
                                elif event == cv2.EVENT_LBUTTONUP:
                                    state['drawing'] = False
                                    if state['current_box_start']:
                                        x1, y1 = state['current_box_start']
                                        x2, y2 = x, y
                                        x1, x2 = min(x1, x2), max(x1, x2)
                                        y1, y2 = min(y1, y2), max(y1, y2)
                                        if x2 - x1 > 5 and y2 - y1 > 5:
                                            current_boxes.append([x1, y1, x2, y2])
                                        state['current_box_start'] = None
                            elif state['interaction_mode'] == 'brush':
                                if event == cv2.EVENT_LBUTTONDOWN:
                                    state['drawing'] = True
                                    cv2.circle(manual_mask_add, (x, y), state['brush_radius'], 1, -1)
                                    cv2.circle(manual_mask_remove, (x, y), state['brush_radius'], 0, -1)
                                elif event == cv2.EVENT_RBUTTONDOWN:
                                    state['drawing'] = True
                                    cv2.circle(manual_mask_remove, (x, y), state['brush_radius'], 1, -1)
                                    cv2.circle(manual_mask_add, (x, y), state['brush_radius'], 0, -1)
                                elif event == cv2.EVENT_MOUSEMOVE:
                                    if state['drawing']:
                                        if flags & cv2.EVENT_FLAG_LBUTTON:
                                            cv2.circle(manual_mask_add, (x, y), state['brush_radius'], 1, -1)
                                            cv2.circle(manual_mask_remove, (x, y), state['brush_radius'], 0, -1)
                                        elif flags & cv2.EVENT_FLAG_RBUTTON:
                                            cv2.circle(manual_mask_remove, (x, y), state['brush_radius'], 1, -1)
                                            cv2.circle(manual_mask_add, (x, y), state['brush_radius'], 0, -1)
                                elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                                    state['drawing'] = False

                        window_name = "Grounding SAM | Mask Review"
                        image_width, image_height = image.size
                        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                        cv2.setMouseCallback(window_name, on_mouse)

                        while True:
                            temp_viz = np.array(image.convert("RGB"))
                            temp_viz = cv2.cvtColor(temp_viz, cv2.COLOR_RGB2BGR)
                            manual_mask_add = ego.exclude(manual_mask_add)
                            manual_mask_remove = ego.exclude(manual_mask_remove)

                            current_mask_preview = np.any(masks > 0, axis=0).copy() if masks.shape[0] > 0 else np.zeros(temp_viz.shape[:2], dtype=bool)
                            current_mask_preview[manual_mask_add > 0] = True
                            current_mask_preview[manual_mask_remove > 0] = False
                            temp_viz = review_overlay(temp_viz, current_mask_preview, ego)

                            temp_viz[manual_mask_remove > 0] = temp_viz[manual_mask_remove > 0] * 0.5 + np.array([0, 0, 128], dtype=np.uint8) * 0.5

                            for i, pt in enumerate(current_user_pts):
                                color = (0, 255, 0) if current_user_lbls[i] == 1 else (0, 0, 255)
                                cv2.circle(temp_viz, (pt[0], pt[1]), 4, color, -1)

                            for box in current_boxes:
                                cv2.rectangle(temp_viz, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                            if state['drawing'] and state['interaction_mode'] == 'box' and state['current_box_start']:
                                cv2.rectangle(temp_viz, state['current_box_start'], (state['ix'], state['iy']), (255, 0, 0), 2)

                            if state['interaction_mode'] == 'brush':
                                cv2.circle(temp_viz, (state['ix'], state['iy']), state['brush_radius'], (255, 255, 255), 1)

                            panel_lines = [
                                "MASK REVIEW",
                                f"Reason: {reason}",
                                "",
                                "[P] Point | [B] Box",
                                "[D] Brush | [C] Clear",
                                "Left: add | Right: remove",
                                "[- / +] Brush size",
                                f"Mode: {state['interaction_mode'].upper()}",
                                f"Brush: {state['brush_radius']} px",
                                "[S] Save | [Q] Skip",
                                "",
                                ego.description,
                                "Cyan: robot / ignored" if ego.mask is not None else "No ego pixels excluded",
                                "Not a negative label",
                                "RGB input stays unchanged",
                            ]
                            display = render_review_panel(temp_viz, panel_lines)

                            cv2.imshow(window_name, display)
                            key = cv2.waitKey(1) & 0xFF

                            if key == ord('p'):
                                state['interaction_mode'] = 'point'
                            elif key == ord('b'):
                                state['interaction_mode'] = 'box'
                            elif key == ord('d'):
                                state['interaction_mode'] = 'brush'
                            elif key == ord('+') or key == ord('='):
                                state['brush_radius'] = min(150, state['brush_radius'] + 5)
                            elif key == ord('-'):
                                state['brush_radius'] = max(2, state['brush_radius'] - 5)
                            elif key == ord('c'):
                                current_user_pts.clear()
                                current_user_lbls.clear()
                                current_boxes.clear()
                                manual_mask_add.fill(0)
                                manual_mask_remove.fill(0)
                            elif key == ord('s'):
                                run_predict = False
                                user_pts_arr = None
                                user_lbls_arr = None

                                if len(current_boxes) > 0:
                                    new_boxes = torch.tensor(current_boxes, dtype=torch.float32).to(input_boxes.device)
                                    input_boxes = torch.cat([input_boxes, new_boxes])
                                    # prior logits cover only the earlier boxes; the expanded prompt batch gets none
                                    logits = None
                                    OBJECTS.extend(["manual"] * len(current_boxes))
                                    run_predict = True

                                if len(current_user_pts) > 0:
                                    user_pts_arr = np.array(current_user_pts)
                                    user_lbls_arr = np.array(current_user_lbls)
                                    run_predict = True

                                if run_predict:
                                    n_objects = input_boxes.shape[0]
                                    if input_boxes.shape[0] == 0 and user_pts_arr is not None:
                                        point_coords_batch = user_pts_arr[None, :, :]
                                        point_labels_batch = user_lbls_arr[None, :]
                                        masks, scores, logits = image_predictor.predict(
                                            point_coords=point_coords_batch,
                                            point_labels=point_labels_batch,
                                            box=None,
                                            multimask_output=False,
                                        )
                                        if masks.ndim == 2:
                                            masks = masks[None]
                                            scores = scores[None]
                                            logits = logits[None]
                                        elif masks.ndim == 4:
                                            masks = masks.squeeze(1)
                                        if masks.shape[0] > 0:
                                            OBJECTS.append("manual")
                                            ys, xs = np.where(masks[0] > 0)
                                            if len(xs) > 0:
                                                input_boxes = torch.tensor(
                                                    [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]],
                                                    dtype=torch.float32,
                                                    device=device,
                                                )
                                    else:
                                        if user_pts_arr is not None:
                                            point_coords_batch = np.tile(user_pts_arr[None, :, :], (n_objects, 1, 1))
                                            point_labels_batch = np.tile(user_lbls_arr[None, :], (n_objects, 1))
                                        else:
                                            point_coords_batch = None
                                            point_labels_batch = None

                                        masks, scores, logits = image_predictor.predict(
                                            point_coords=point_coords_batch,
                                            point_labels=point_labels_batch,
                                            box=input_boxes,
                                            mask_input=logits if point_coords_batch is not None else None,
                                            multimask_output=False,
                                        )
                                        if masks.ndim == 4:
                                            masks = masks.squeeze(1)

                                if np.any(manual_mask_add) or np.any(manual_mask_remove):
                                    if masks.shape[0] == 0:
                                        ys, xs = np.where(manual_mask_add > 0)
                                        if len(xs) > 0:
                                            masks = manual_mask_add.astype(bool)[None]
                                            OBJECTS.append("manual")
                                            box = torch.tensor(
                                                [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]],
                                                dtype=torch.float32,
                                                device=input_boxes.device,
                                            )
                                            input_boxes = torch.cat([input_boxes, box])
                                    for i in range(masks.shape[0]):
                                        masks[i, manual_mask_add > 0] = True
                                        masks[i, manual_mask_remove > 0] = False

                                break
                            elif key == ord('q'):
                                break

                        cv2.destroyWindow(window_name)
                masks = ego.exclude(masks)

                if config["processing"].get("human_in_the_loop", False):
                     previous_step_mask = np.any(masks > 0, axis=0).astype(np.uint8)

                if mask_dict.promote_type == "mask":
                    new_masks_tensor = torch.tensor(masks).to(device)

                    mask_dict.add_new_frame_annotation(
                        mask_list=new_masks_tensor,
                        box_list=input_boxes,
                        label_list=OBJECTS
                    )

                objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.95, objects_count=objects_count)
            else:
                mask_dict = sam2_masks

            if len(mask_dict.labels) == 0:
                raise RuntimeError(
                    f"No terrain detections for {first_frame_in_step}; labels are incomplete. "
                    "Review the prompts and detection thresholds before rerunning."
                )
            else:
                video_predictor.reset_state(inference_state)

                for object_id, object_info in mask_dict.labels.items():
                    frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                            inference_state,
                            local_start_idx,
                            object_id,
                            object_info.mask,
                        )

                propagation_gen = video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=local_start_idx)

                while True:
                    try:
                        out_frame_idx, out_obj_ids, out_mask_logits = next(propagation_gen)
                    except StopIteration:
                        break

                    frame_masks = MaskDictionaryModel()

                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > -1.0)
                        if ego.mask is not None:
                            if ego_valid_tensor is None or ego_valid_tensor.device != out_mask.device:
                                ego_valid_tensor = torch.as_tensor(~ego.mask, device=out_mask.device)
                            out_mask = out_mask & ego_valid_tensor
                        object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                        object_info.update_box()
                        frame_masks.labels[out_obj_id] = object_info

                        if out_frame_idx in temp_to_original_mapping:
                                real_image_name = temp_to_original_mapping[out_frame_idx]
                                image_base_name = os.path.splitext(real_image_name)[0]
                                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                                frame_masks.mask_height = out_mask.shape[-2]
                                frame_masks.mask_width = out_mask.shape[-1]
                        else:
                                continue

                    sam2_masks = copy.deepcopy(frame_masks)
                    if out_frame_idx not in temp_to_original_mapping:
                         continue

                    original_fname = temp_to_original_mapping[out_frame_idx]

                    frame_masks.mask_name = f"mask_{os.path.splitext(original_fname)[0]}.npy"

                    mask_map = frame_masks.labels
                    positive_mask_img = torch.zeros(frame_masks.mask_height, frame_masks.mask_width)
                    binary_accumulator = np.zeros((frame_masks.mask_height, frame_masks.mask_width), dtype=np.uint8)

                    for obj_id, obj_info in mask_map.items():
                        class_name = obj_info.class_name.lower().strip()
                        is_positive = False
                        mode = config["prompts"]["mode"]

                        if mode == "positive":
                            is_positive = class_name == "manual" or any(p in class_name for p in positive_prompts) or any(class_name in p for p in positive_prompts)

                        mask_bool = obj_info.mask == True

                        if is_positive:
                            positive_mask_img[mask_bool] = obj_id
                            binary_accumulator[mask_bool.cpu().numpy()] = 255


                    if config["prompts"]["mode"] == "positive":
                        binary_accumulator = CommonUtils.remove_small_islands(
                            binary_accumulator,
                            max_area=config["processing"].get("max_island_size", 3000)
                        )

                        binary_accumulator = CommonUtils.fill_small_holes(
                            binary_accumulator,
                            max_area=config["processing"].get("max_hole_size", 15000)
                        )

                        binary_accumulator = CommonUtils.fill_surgical_gaps(binary_accumulator)
                        binary_accumulator = ego.exclude(binary_accumulator)

                        positive_mask_img[torch.from_numpy(binary_accumulator == 0)] = 0

                        total_pixels = frame_masks.mask_height * frame_masks.mask_width
                        occupancy_ratio = np.sum(binary_accumulator > 0) / total_pixels
                        max_occupancy = config["processing"].get("max_occupancy_ratio", 1.0)

                        if occupancy_ratio > max_occupancy:
                            binary_accumulator[:] = 0
                            positive_mask_img[:] = 0
                    if ego.mask is not None:
                        positive_mask_img[torch.from_numpy(ego.mask)] = 0

                    if config["processing"].get("save_npy", False):
                        positive_mask_numpy = positive_mask_img.numpy().astype(np.uint16)
                        np.save(os.path.join(positive_mask_dir, frame_masks.mask_name), positive_mask_numpy)

                    pos_mask_name = f"{positive_mask_prefix}{os.path.splitext(original_fname)[0]}.tif"

                    if config["prompts"]["mode"] == "positive":
                        pos_mask_np = (positive_mask_img.cpu().numpy() > 0).astype(np.uint8) * 255
                        cv2.imwrite(os.path.join(positive_mask_dir, pos_mask_name), pos_mask_np)

                    if config["processing"].get("save_visualization", False):
                        img_full_path = os.path.join(video_dir, "rgb", original_fname)
                        viz_img = cv2.imread(img_full_path)

                        if viz_img is not None:

                            viz_dir = os.path.join(output_dir, "visualization")
                            if not os.path.exists(viz_dir): os.makedirs(viz_dir)

                            if config["prompts"]["mode"] == "positive" and pos_mask_np is not None:
                                overlay_pos = viz_img.copy()
                                overlay_pos[pos_mask_np > 0] = (0, 255, 0)

                                cv2.addWeighted(overlay_pos, 0.7, viz_img, 0.3, 0, overlay_pos)
                                overlay_pos = ego.overlay(overlay_pos)

                                if out_frame_idx == local_start_idx and points_batch is not None:
                                     for box_idx in range(points_batch.shape[0]):
                                         pts = points_batch[box_idx]
                                         for pt in pts:
                                             px, py = int(pt[0]), int(pt[1])
                                             if px == 0 and py == 0: continue
                                             cv2.circle(overlay_pos, (px, py), 4, (255, 255, 0), -1)

                                viz_name_pos = original_fname.replace("rgb_", "vis_").replace(".png", ".jpg").replace(".tif", ".jpg")
                                cv2.imwrite(os.path.join(viz_dir, viz_name_pos), overlay_pos)

                    if config["processing"].get("save_json", False):
                        json_data = frame_masks.to_dict()
                        json_data_path = os.path.join(json_data_dir, frame_masks.mask_name.replace(".npy", ".json"))
                        with open(json_data_path, "w") as f:
                            json.dump(json_data, f)

        try:
            video_predictor.reset_state(inference_state)
        except Exception:
            pass

        del inference_state
        gc.collect()
        torch.cuda.empty_cache()

        shutil.rmtree(temp_batch_dir)


    if os.path.exists(temp_batch_dir):
        try:
            shutil.rmtree(temp_batch_dir)
        except Exception:
            pass

    all_labeled = update_positive_mask_references(output_dir, config, allow_partial=True)
    if all_labeled:
        with open(os.path.join(output_dir, "sam_completed.txt"), "w") as f:
            f.write("done")

def main() -> None:
    """
    Run Grounding SAM labeling on the datasets given by --dataset.
    """
    parser = argparse.ArgumentParser(description="Grounding SAM Segmentation")
    parser.add_argument("--config", type=str, default=None, help="YAML overrides for the generation configuration")
    parser.add_argument("--dataset", type=str, nargs="+", required=True, help="Paths to processed dataset directories")
    parser.add_argument("--box_threshold", type=float, help="Override box threshold")
    parser.add_argument("--text_threshold", type=float, help="Override text threshold")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--step_positive", type=int, help="Override step positive")
    parser.add_argument("--hil", type=str, help="Override Human in the Loop (true/false)")
    parser.add_argument("--resume", type=str, help="Resume from last processed image (true/false)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.box_threshold is not None:
        config["grounding_dino"]["box_threshold"] = args.box_threshold
    if args.text_threshold is not None:
        config["grounding_dino"]["text_threshold"] = args.text_threshold
    if args.batch_size is not None:
        config["processing"]["batch_size"] = int(args.batch_size)
    if args.step_positive is not None:
        config["grounding_dino"]["step_positive"] = args.step_positive
    if args.hil is not None:
        config["processing"]["human_in_the_loop"] = args.hil.lower() == "true"

    dataset_dirs_to_process = args.dataset

    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = config["sam2"]["checkpoint"]
    model_cfg = config["sam2"]["config"]
    device = config["device"] if torch.cuda.is_available() else "cpu"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device=device))

    model_id = config["grounding_dino"]["model_id"]
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    positive_prompts = [p.lower() for p in config["prompts"]["positive"]]
    text = " . ".join(positive_prompts) + "."

    for dataset_dir in dataset_dirs_to_process:
        process_dataset(dataset_dir, config, device, video_predictor, image_predictor,
                        processor, grounding_model, text, positive_prompts, args.resume)

if __name__ == "__main__":
    main()
