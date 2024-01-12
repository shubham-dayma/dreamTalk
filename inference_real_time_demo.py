import argparse
import json
import os
import shutil
import numpy as np
import torch
from scipy.io import loadmat

from configs.default import get_cfg_defaults
from core.networks.diffusion_net import DiffusionNet
from core.networks.diffusion_util import NoisePredictor, VarianceSchedule
from core.utils import (
    crop_src_image,
    get_pose_params,
    get_video_style_clip,
    get_wav2vec_audio_window,
)
from generators.utils import get_netG

import cv2
from PIL import Image
import torchvision
import torchvision.transforms as transforms

wav_path="data/audio/acknowledgement_english.m4a"
image_path="data/src_img/uncropped/male_face.png"
img_crop=True
style_clip_path="data/style_clip/3DMM/M030_front_neutral_level1_001.mat"
pose_path="data/pose/RichardShelby_front_neutral_level1_001.mat"
max_gen_len=1000
cfg_scale=1.0
output_name="acknowledgement_english@M030_front_neutral_level1_001@male_face"
device="cuda"

@torch.no_grad()
def get_diff_net(cfg, device):
    diff_net = DiffusionNet(
        cfg=cfg,
        net=NoisePredictor(cfg),
        var_sched=VarianceSchedule(
            num_steps=cfg.DIFFUSION.SCHEDULE.NUM_STEPS,
            beta_1=cfg.DIFFUSION.SCHEDULE.BETA_1,
            beta_T=cfg.DIFFUSION.SCHEDULE.BETA_T,
            mode=cfg.DIFFUSION.SCHEDULE.MODE,
        ),
    )
    checkpoint = torch.load(cfg.INFERENCE.CHECKPOINT, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    diff_net_dict = {
        k[9:]: v for k, v in model_state_dict.items() if k[:9] == "diff_net."
    }
    diff_net.load_state_dict(diff_net_dict, strict=True)
    diff_net.eval()

    return diff_net

@torch.no_grad()
def inference_one_video(
    cfg,
    audio_path,
    style_clip_path,
    pose_path,
    output_path,
    diff_net,
    device,
    max_audio_len=None,
    sample_method="ddim",
    ddim_num_step=10,
):
    audio_raw = audio_data = np.load(audio_path)

    if max_audio_len is not None:
        audio_raw = audio_raw[: max_audio_len * 50]
    gen_num_frames = len(audio_raw) // 2

    audio_win_array = get_wav2vec_audio_window(
        audio_raw,
        start_idx=0,
        num_frames=gen_num_frames,
        win_size=cfg.WIN_SIZE,
    )

    audio_win = torch.tensor(audio_win_array).to(device)
    audio = audio_win.unsqueeze(0)

    # the second parameter is "" because of bad interface design...
    style_clip_raw, style_pad_mask_raw = get_video_style_clip(
        style_clip_path, "", style_max_len=256, start_idx=0
    )

    style_clip = style_clip_raw.unsqueeze(0).to(device)
    style_pad_mask = (
        style_pad_mask_raw.unsqueeze(0).to(device)
        if style_pad_mask_raw is not None
        else None
    )

    gen_exp_stack = diff_net.sample(
        audio,
        style_clip,
        style_pad_mask,
        output_dim=cfg.DATASET.FACE3D_DIM,
        use_cf_guidance=cfg.CF_GUIDANCE.INFERENCE,
        cfg_scale=cfg.CF_GUIDANCE.SCALE,
        sample_method=sample_method,
        ddim_num_step=ddim_num_step,
    )
    gen_exp = gen_exp_stack[0].cpu().numpy()

    pose_ext = pose_path[-3:]
    pose = None
    pose = get_pose_params(pose_path)
    # (L, 9)

    selected_pose = None
    if len(pose) >= len(gen_exp):
        selected_pose = pose[: len(gen_exp)]
    else:
        selected_pose = pose[-1].unsqueeze(0).repeat(len(gen_exp), 1)
        selected_pose[: len(pose)] = pose

    gen_exp_pose = np.concatenate((gen_exp, selected_pose), axis=1)
    np.save(output_path, gen_exp_pose)
    return output_path

def obtain_seq_index(index, num_frames, radius):
    seq = list(range(index - radius, index + radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq

@torch.no_grad()
def render_video(
    net_G,
    src_img_path,
    exp_path,
    output_path,
    device,
    silent=False,
    semantic_radius=13,
    fps=30,
    split_size=16,
    no_move=False,
):
    """
    exp: (N, 73)
    """
    target_exp_seq = np.load(exp_path)
    if target_exp_seq.shape[1] == 257:
        exp_coeff = target_exp_seq[:, 80:144]
        angle_trans_crop = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9370641, 126.84911, 129.03864],
            dtype=np.float32,
        )
        target_exp_seq = np.concatenate(
            [exp_coeff, angle_trans_crop[None, ...].repeat(exp_coeff.shape[0], axis=0)],
            axis=1,
        )
        # (L, 73)
    elif target_exp_seq.shape[1] == 73:
        if no_move:
            target_exp_seq[:, 64:] = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9370641, 126.84911, 129.03864],
                dtype=np.float32,
            )
    else:
        raise NotImplementedError

    frame = cv2.imread(src_img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    src_img_raw = Image.fromarray(frame)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    src_img = image_transform(src_img_raw)

    target_win_exps = []
    for frame_idx in range(len(target_exp_seq)):
        win_indices = obtain_seq_index(
            frame_idx, target_exp_seq.shape[0], semantic_radius
        )
        win_exp = torch.tensor(target_exp_seq[win_indices]).permute(1, 0)
        # (73, 27)
        target_win_exps.append(win_exp)

    target_exp_concat = torch.stack(target_win_exps, dim=0)
    target_splited_exps = torch.split(target_exp_concat, split_size, dim=0)
    output_imgs = []
    for win_exp in target_splited_exps:
        win_exp = win_exp.to(device)
        cur_src_img = src_img.expand(win_exp.shape[0], -1, -1, -1).to(device)
        output_dict = net_G(cur_src_img, win_exp)
        output_imgs.append(output_dict["fake_image"].cpu().clamp_(-1, 1))

    output_imgs = torch.cat(output_imgs, 0)
    transformed_imgs = ((output_imgs + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(output_path, transformed_imgs.cpu(), fps)
    
if __name__ == "__main__":
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, set --device=cpu to use CPU.")
        exit(1)

    device = torch.device(device)

    cfg = get_cfg_defaults()
    cfg.CF_GUIDANCE.SCALE = cfg_scale
    cfg.freeze()

    tmp_dir = f"tmp/{output_name}"
    os.makedirs(tmp_dir, exist_ok=True)

    audio_feat_path = os.path.join(tmp_dir, f"{output_name}_wav2vec.npy")
    
    # get src image
    src_img_path = os.path.join(tmp_dir, "src_img.png")
    if img_crop:
        crop_src_image(image_path, src_img_path, 0.4)
    else:
        shutil.copy(image_path, src_img_path)

    with torch.no_grad():
        # get diff model and load checkpoint
        diff_net = get_diff_net(cfg, device).to(device)
        # generate face motion
        face_motion_path = os.path.join(tmp_dir, f"{output_name}_facemotion.npy")
        inference_one_video(
            cfg,
            audio_feat_path,
            style_clip_path,
            pose_path,
            face_motion_path,
            diff_net,
            device,
            max_audio_len=max_gen_len,
        )
        # get renderer
        renderer = get_netG("checkpoints/renderer.pt", device)
        # render video
        output_video_path = f"output_video/{output_name}.mp4"
        render_video(
            renderer,
            src_img_path,
            face_motion_path,
            output_video_path,
            device,
            fps=25,
            no_move=False,
        )