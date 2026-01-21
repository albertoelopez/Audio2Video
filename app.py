#!/usr/bin/env python3
import gradio as gr
import os
import json
import time
import uuid
import subprocess
import requests
import websocket
import shutil
import threading
from pathlib import Path
from datetime import datetime

COMFYUI_URL = "http://127.0.0.1:8188"
BASE_DIR = Path(__file__).parent
PROJECTS_DIR = BASE_DIR / "projects"
COMFYUI_DIR = BASE_DIR / "ComfyUI"
HEARTLIB_DIR = BASE_DIR / "heartlib"

PROJECTS_DIR.mkdir(exist_ok=True)


class ComfyUIClient:
    def __init__(self, url=COMFYUI_URL):
        self.url = url
        self.client_id = str(uuid.uuid4())

    def is_running(self):
        try:
            requests.get(f"{self.url}/system_stats", timeout=2)
            return True
        except:
            return False

    def queue_prompt(self, workflow):
        payload = {"prompt": workflow, "client_id": self.client_id}
        response = requests.post(f"{self.url}/prompt", json=payload)
        return response.json()

    def get_history(self, prompt_id):
        response = requests.get(f"{self.url}/history/{prompt_id}")
        return response.json()

    def upload_file(self, file_path, subfolder="", file_type="input"):
        file_path = Path(file_path)
        with open(file_path, "rb") as f:
            files = {"image": (file_path.name, f, "application/octet-stream")}
            data = {"subfolder": subfolder, "type": file_type}
            response = requests.post(f"{self.url}/upload/image", files=files, data=data)
        return response.json()

    def wait_for_completion(self, prompt_id, timeout=600, progress_callback=None):
        ws_url = f"ws://127.0.0.1:8188/ws?clientId={self.client_id}"
        try:
            ws = websocket.create_connection(ws_url, timeout=10)
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            return False

        start_time = time.time()
        try:
            while time.time() - start_time < timeout:
                try:
                    msg = ws.recv()
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == "executing":
                            exec_data = data.get("data", {})
                            if exec_data.get("prompt_id") == prompt_id:
                                if exec_data.get("node") is None:
                                    return True
                        elif data.get("type") == "progress":
                            prog = data.get("data", {})
                            if progress_callback:
                                progress_callback(prog.get("value", 0), prog.get("max", 100))
                except websocket.WebSocketTimeoutException:
                    continue
        finally:
            ws.close()
        return False

    def get_output_video(self, prompt_id):
        history = self.get_history(prompt_id)
        if prompt_id not in history:
            return None
        outputs = history[prompt_id].get("outputs", {})
        for node_id, node_output in outputs.items():
            for key in ["gifs", "videos"]:
                if key in node_output:
                    for video in node_output[key]:
                        return video.get("filename")
        return None


comfy = ComfyUIClient()


def run_command(cmd, capture=True, cwd=None):
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
            return result.returncode == 0, result.stdout + result.stderr
        else:
            result = subprocess.run(cmd, cwd=cwd)
            return result.returncode == 0, ""
    except Exception as e:
        return False, str(e)


def create_project(name):
    project_dir = PROJECTS_DIR / name
    for subdir in ["input", "audio", "video"]:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)
    return project_dir


def generate_music_heartmula(lyrics, tags, project_dir, progress=gr.Progress()):
    progress(0.1, desc="Starting HeartMuLa...")

    lyrics_file = project_dir / "input" / "lyrics.txt"
    tags_file = project_dir / "input" / "tags.txt"
    output_file = project_dir / "audio" / "full_track.mp3"

    lyrics_file.write_text(lyrics)
    tags_file.write_text(tags)

    progress(0.3, desc="Generating music...")

    success, output = run_command([
        "python", str(HEARTLIB_DIR / "examples" / "run_music_generation.py"),
        "--model_path", str(HEARTLIB_DIR / "ckpt"),
        "--lyrics", str(lyrics_file),
        "--tags", str(tags_file),
        "--save_path", str(output_file),
        "--version", "3B",
        "--max_audio_length_ms", "240000"
    ], cwd=str(HEARTLIB_DIR))

    if success and output_file.exists():
        progress(1.0, desc="Done!")
        return str(output_file), "Music generated successfully!"
    else:
        return None, f"Error: {output}"


def separate_stems(audio_path, project_dir, progress=gr.Progress()):
    progress(0.2, desc="Running Demucs...")

    audio_path = Path(audio_path)
    stems_dir = project_dir / "audio" / "stems"

    success, output = run_command([
        "demucs", "--two-stems=vocals",
        "-o", str(stems_dir),
        str(audio_path)
    ])

    if not success:
        return None, None, f"Demucs error: {output}"

    progress(0.8, desc="Moving files...")

    stem_dir = stems_dir / "htdemucs" / audio_path.stem
    vocals_src = stem_dir / "vocals.wav"
    instrumental_src = stem_dir / "no_vocals.wav"

    vocals_dst = project_dir / "audio" / "vocals.wav"
    instrumental_dst = project_dir / "audio" / "instrumental.wav"

    if vocals_src.exists():
        shutil.move(str(vocals_src), str(vocals_dst))
    if instrumental_src.exists():
        shutil.move(str(instrumental_src), str(instrumental_dst))

    progress(1.0, desc="Done!")

    vocals_result = str(vocals_dst) if vocals_dst.exists() else None
    instrumental_result = str(instrumental_dst) if instrumental_dst.exists() else None

    return vocals_result, instrumental_result, "Stems separated!"


def transcribe_lyrics(audio_path, project_dir, use_heartmula=False, progress=gr.Progress()):
    progress(0.2, desc="Transcribing...")

    srt_path = project_dir / "lyrics.srt"

    if use_heartmula:
        json_path = project_dir / "lyrics.json"
        success, output = run_command([
            "python", "-m", "heartlib.transcribe",
            "--audio_path", str(audio_path),
            "--output_path", str(json_path)
        ], cwd=str(HEARTLIB_DIR.parent))
        if success and json_path.exists():
            json_to_srt(json_path, srt_path)
    else:
        success, output = run_command([
            "whisper", str(audio_path),
            "--model", "large-v3",
            "--word_timestamps", "True",
            "--output_format", "srt",
            "--output_dir", str(project_dir)
        ])
        whisper_out = project_dir / f"{Path(audio_path).stem}.srt"
        if whisper_out.exists():
            shutil.move(str(whisper_out), str(srt_path))

    if srt_path.exists():
        progress(1.0, desc="Done!")
        return str(srt_path), srt_path.read_text(), "Transcription complete!"
    else:
        return None, "", f"Error: {output}"


def json_to_srt(json_path, srt_path):
    with open(json_path) as f:
        data = json.load(f)

    with open(srt_path, "w") as out:
        for i, seg in enumerate(data.get("segments", []), 1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            text = seg["text"].strip()
            out.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def get_audio_duration(audio_path):
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ], capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 60.0


def split_audio_sections(audio_path, project_dir, section_length=8):
    duration = get_audio_duration(audio_path)
    sections = []

    start = 0
    idx = 1
    while start < duration:
        end = min(start + section_length, duration)
        section_path = project_dir / "audio" / f"section_{idx:03d}.wav"

        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start), "-t", str(end - start),
            "-acodec", "pcm_s16le", "-ar", "44100",
            str(section_path)
        ], capture_output=True)

        sections.append({
            "path": section_path,
            "start": start,
            "end": end,
            "duration": end - start,
            "index": idx
        })

        start = end
        idx += 1

    return sections


def build_wan_workflow(audio_file, image_file, prompt, negative_prompt,
                       frames=120, width=832, height=480, steps=20, cfg=6.0, seed=-1):
    if seed == -1:
        seed = int(time.time() * 1000) % (2**32)

    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "wan2.2_s2v_14B_fp8_scaled.safetensors",
            "weight_dtype": "fp8_e4m3fn"
        }},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "sd3"
        }},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": "wan_2.1_vae.safetensors"
        }},
        "4": {"class_type": "AudioEncoderLoader", "inputs": {
            "audio_encoder_name": "wav2vec2_large_english_fp16.safetensors"
        }},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_file}},
        "6": {"class_type": "LoadAudio", "inputs": {"audio": audio_file}},
        "7": {"class_type": "AudioEncoderEncode", "inputs": {
            "audio_encoder": ["4", 0],
            "audio": ["6", 0]
        }},
        "8": {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt, "clip": ["2", 0]
        }},
        "9": {"class_type": "CLIPTextEncode", "inputs": {
            "text": negative_prompt, "clip": ["2", 0]
        }},
        "10": {"class_type": "WanSoundImageToVideo", "inputs": {
            "positive": ["8", 0],
            "negative": ["9", 0],
            "vae": ["3", 0],
            "width": width,
            "height": height,
            "length": frames,
            "batch_size": 1,
            "audio_encoder_output": ["7", 0],
            "ref_image": ["5", 0]
        }},
        "11": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0],
            "positive": ["10", 0],
            "negative": ["10", 1],
            "latent_image": ["10", 2],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0
        }},
        "12": {"class_type": "VAEDecode", "inputs": {
            "samples": ["11", 0], "vae": ["3", 0]
        }},
        "13": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["12", 0], "frame_rate": 24, "loop_count": 0,
            "filename_prefix": "wan_s2v", "format": "video/h264-mp4", "save_output": True
        }}
    }


def generate_video_clips(vocals_path, reference_image, project_dir, prompt, negative_prompt,
                         steps, cfg, progress=gr.Progress()):
    if not comfy.is_running():
        return [], "ComfyUI not running! Start it first."

    progress(0.05, desc="Splitting audio...")
    sections = split_audio_sections(vocals_path, project_dir, section_length=8)

    progress(0.1, desc="Uploading reference image...")
    img_result = comfy.upload_file(reference_image)
    image_name = img_result.get("name", Path(reference_image).name)

    generated_clips = []
    base_seed = int(time.time()) % 1000000

    for i, section in enumerate(sections):
        idx = section["index"]
        prog = 0.1 + (0.85 * i / len(sections))
        progress(prog, desc=f"Generating clip {idx}/{len(sections)}...")

        audio_result = comfy.upload_file(section["path"])
        audio_name = audio_result.get("name", section["path"].name)

        frames = min(int(section["duration"] * 24), 192)

        workflow = build_wan_workflow(
            audio_file=audio_name,
            image_file=image_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            frames=frames,
            steps=steps,
            cfg=cfg,
            seed=base_seed + idx
        )

        result = comfy.queue_prompt(workflow)
        prompt_id = result.get("prompt_id")

        if not prompt_id:
            continue

        success = comfy.wait_for_completion(prompt_id, timeout=600)

        if success:
            output_name = comfy.get_output_video(prompt_id)
            if output_name:
                comfy_output = COMFYUI_DIR / "output" / output_name
                clip_path = project_dir / "video" / f"clip_{idx:03d}.mp4"

                if comfy_output.exists():
                    shutil.copy(str(comfy_output), str(clip_path))
                    generated_clips.append(str(clip_path))

    progress(1.0, desc="Done!")
    return generated_clips, f"Generated {len(generated_clips)}/{len(sections)} clips"


def assemble_final_video(project_dir, audio_path, srt_path, font_size, progress=gr.Progress()):
    progress(0.2, desc="Building clip list...")

    video_dir = project_dir / "video"
    clips = sorted(video_dir.glob("clip_*.mp4"))

    if not clips:
        return None, "No video clips found!"

    clips_file = project_dir / "clips.txt"
    with open(clips_file, "w") as f:
        for clip in clips:
            f.write(f"file '{clip.absolute()}'\n")

    output = project_dir / "final_video.mp4"

    style = (
        f"FontSize={font_size},"
        "FontName=Arial Bold,"
        "PrimaryColour=&HFFFFFF,"
        "OutlineColour=&H000000,"
        "Outline=2,"
        "Shadow=1,"
        "MarginV=50"
    )

    progress(0.5, desc="Encoding final video...")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(clips_file),
        "-i", str(audio_path),
        "-vf", f"subtitles={srt_path}:force_style='{style}'",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output)
    ]

    success, out = run_command(cmd)

    if success and output.exists():
        progress(1.0, desc="Done!")
        return str(output), "Final video assembled!"
    else:
        return None, f"Error: {out}"


def run_full_pipeline_create(project_name, lyrics, tags, reference_image,
                             prompt, negative_prompt, steps, cfg, font_size,
                             progress=gr.Progress()):
    logs = []

    def log(msg):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        return "\n".join(logs)

    yield None, None, log("Starting pipeline...")

    project_dir = create_project(project_name)

    if reference_image:
        ref_path = project_dir / "input" / "reference.png"
        shutil.copy(reference_image, ref_path)
    else:
        yield None, None, log("ERROR: Reference image required!")
        return

    yield None, None, log("Generating music with HeartMuLa...")
    audio_path, msg = generate_music_heartmula(lyrics, tags, project_dir, progress)
    if not audio_path:
        yield None, None, log(f"ERROR: {msg}")
        return
    yield audio_path, None, log(f"Music generated: {audio_path}")

    yield audio_path, None, log("Separating stems...")
    vocals, instrumental, msg = separate_stems(audio_path, project_dir, progress)
    if not vocals:
        yield audio_path, None, log(f"ERROR: {msg}")
        return
    yield audio_path, None, log("Stems separated!")

    yield audio_path, None, log("Transcribing lyrics...")
    srt_path, srt_text, msg = transcribe_lyrics(vocals, project_dir, use_heartmula=False, progress=progress)
    if not srt_path:
        yield audio_path, None, log(f"ERROR: {msg}")
        return
    yield audio_path, None, log("Lyrics transcribed!")

    yield audio_path, None, log("Generating video clips (this takes a while)...")
    clips, msg = generate_video_clips(vocals, ref_path, project_dir, prompt, negative_prompt, steps, cfg, progress)
    yield audio_path, None, log(msg)

    if not clips:
        yield audio_path, None, log("ERROR: No clips generated!")
        return

    yield audio_path, None, log("Assembling final video...")
    final_video, msg = assemble_final_video(project_dir, audio_path, srt_path, font_size, progress)

    if final_video:
        yield audio_path, final_video, log(f"DONE! Output: {final_video}")
    else:
        yield audio_path, None, log(f"ERROR: {msg}")


def run_full_pipeline_upload(project_name, audio_file, reference_image,
                             prompt, negative_prompt, steps, cfg, font_size,
                             progress=gr.Progress()):
    logs = []

    def log(msg):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        return "\n".join(logs)

    yield None, None, log("Starting pipeline...")

    project_dir = create_project(project_name)

    if not audio_file:
        yield None, None, log("ERROR: Audio file required!")
        return

    if not reference_image:
        yield None, None, log("ERROR: Reference image required!")
        return

    audio_path = project_dir / "audio" / "full_track.mp3"
    shutil.copy(audio_file, audio_path)

    ref_path = project_dir / "input" / "reference.png"
    shutil.copy(reference_image, ref_path)

    yield str(audio_path), None, log("Files imported!")

    yield str(audio_path), None, log("Separating stems...")
    vocals, instrumental, msg = separate_stems(audio_path, project_dir, progress)
    if not vocals:
        yield str(audio_path), None, log(f"ERROR: {msg}")
        return
    yield str(audio_path), None, log("Stems separated!")

    yield str(audio_path), None, log("Transcribing lyrics...")
    srt_path, srt_text, msg = transcribe_lyrics(vocals, project_dir, use_heartmula=False, progress=progress)
    if not srt_path:
        yield str(audio_path), None, log(f"ERROR: {msg}")
        return
    yield str(audio_path), None, log("Lyrics transcribed!")

    yield str(audio_path), None, log("Generating video clips (this takes a while)...")
    clips, msg = generate_video_clips(vocals, ref_path, project_dir, prompt, negative_prompt, steps, cfg, progress)
    yield str(audio_path), None, log(msg)

    if not clips:
        yield str(audio_path), None, log("ERROR: No clips generated!")
        return

    yield str(audio_path), None, log("Assembling final video...")
    final_video, msg = assemble_final_video(project_dir, audio_path, srt_path, font_size, progress)

    if final_video:
        yield str(audio_path), final_video, log(f"DONE! Output: {final_video}")
    else:
        yield str(audio_path), None, log(f"ERROR: {msg}")


def check_comfyui_status():
    if comfy.is_running():
        return "ComfyUI is running"
    else:
        return "ComfyUI not detected - start it first!"


def get_project_list():
    return [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]


with gr.Blocks(title="AI Music Video Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # AI Music Video Generator
    Create music videos with AI - either generate music from scratch or upload your own track.
    """)

    with gr.Row():
        comfyui_status = gr.Textbox(label="ComfyUI Status", value=check_comfyui_status(), interactive=False)
        refresh_btn = gr.Button("Refresh", size="sm")
        refresh_btn.click(check_comfyui_status, outputs=comfyui_status)

    with gr.Tabs():
        with gr.Tab("Create Music + Video"):
            gr.Markdown("### Generate music from lyrics, then create a synced music video")

            with gr.Row():
                with gr.Column(scale=1):
                    create_project_name = gr.Textbox(
                        label="Project Name",
                        value=f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                    create_lyrics = gr.Textbox(
                        label="Lyrics",
                        placeholder="[verse]\nYour lyrics here...\n\n[chorus]\nChorus lyrics...",
                        lines=10
                    )

                    create_tags = gr.Textbox(
                        label="Style Tags",
                        placeholder="pluggnb, trap, 808 bass, melodic vocals, 140 bpm",
                        lines=2
                    )

                    create_reference = gr.Image(
                        label="Reference Image",
                        type="filepath"
                    )

                with gr.Column(scale=1):
                    create_prompt = gr.Textbox(
                        label="Video Prompt",
                        value="person singing, emotional expression, cinematic lighting, music video aesthetic, 4K",
                        lines=3
                    )

                    create_negative = gr.Textbox(
                        label="Negative Prompt",
                        value="blurry, distorted face, extra limbs, bad anatomy, static, low quality",
                        lines=2
                    )

                    with gr.Row():
                        create_steps = gr.Slider(4, 30, value=20, step=1, label="Steps")
                        create_cfg = gr.Slider(1, 15, value=6, step=0.5, label="CFG")

                    create_font_size = gr.Slider(16, 72, value=36, step=2, label="Lyric Font Size")

            create_btn = gr.Button("Generate Music Video", variant="primary", size="lg")

            with gr.Row():
                create_audio_output = gr.Audio(label="Generated Audio")
                create_video_output = gr.Video(label="Final Video")

            create_logs = gr.Textbox(label="Logs", lines=10, interactive=False)

            create_btn.click(
                run_full_pipeline_create,
                inputs=[
                    create_project_name, create_lyrics, create_tags, create_reference,
                    create_prompt, create_negative, create_steps, create_cfg, create_font_size
                ],
                outputs=[create_audio_output, create_video_output, create_logs]
            )

        with gr.Tab("Upload Music + Video"):
            gr.Markdown("### Upload your own track and generate a synced music video")

            with gr.Row():
                with gr.Column(scale=1):
                    upload_project_name = gr.Textbox(
                        label="Project Name",
                        value=f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                    upload_audio = gr.Audio(
                        label="Upload Audio (MP3/WAV)",
                        type="filepath"
                    )

                    upload_reference = gr.Image(
                        label="Reference Image",
                        type="filepath"
                    )

                with gr.Column(scale=1):
                    upload_prompt = gr.Textbox(
                        label="Video Prompt",
                        value="person singing, emotional expression, cinematic lighting, music video aesthetic, 4K",
                        lines=3
                    )

                    upload_negative = gr.Textbox(
                        label="Negative Prompt",
                        value="blurry, distorted face, extra limbs, bad anatomy, static, low quality",
                        lines=2
                    )

                    with gr.Row():
                        upload_steps = gr.Slider(4, 30, value=20, step=1, label="Steps")
                        upload_cfg = gr.Slider(1, 15, value=6, step=0.5, label="CFG")

                    upload_font_size = gr.Slider(16, 72, value=36, step=2, label="Lyric Font Size")

            upload_btn = gr.Button("Generate Music Video", variant="primary", size="lg")

            with gr.Row():
                upload_audio_output = gr.Audio(label="Processed Audio")
                upload_video_output = gr.Video(label="Final Video")

            upload_logs = gr.Textbox(label="Logs", lines=10, interactive=False)

            upload_btn.click(
                run_full_pipeline_upload,
                inputs=[
                    upload_project_name, upload_audio, upload_reference,
                    upload_prompt, upload_negative, upload_steps, upload_cfg, upload_font_size
                ],
                outputs=[upload_audio_output, upload_video_output, upload_logs]
            )

        with gr.Tab("Step-by-Step"):
            gr.Markdown("### Run individual pipeline steps")

            with gr.Accordion("1. Generate Music (HeartMuLa)", open=False):
                with gr.Row():
                    step_lyrics = gr.Textbox(label="Lyrics", lines=5)
                    step_tags = gr.Textbox(label="Tags", lines=2)
                step_music_project = gr.Textbox(label="Project Name", value="step_project")
                step_music_btn = gr.Button("Generate Music")
                step_music_output = gr.Audio(label="Output")
                step_music_status = gr.Textbox(label="Status")

                def step_generate_music(lyrics, tags, project_name, progress=gr.Progress()):
                    project_dir = create_project(project_name)
                    return generate_music_heartmula(lyrics, tags, project_dir, progress)

                step_music_btn.click(
                    step_generate_music,
                    inputs=[step_lyrics, step_tags, step_music_project],
                    outputs=[step_music_output, step_music_status]
                )

            with gr.Accordion("2. Separate Stems", open=False):
                step_stems_audio = gr.Audio(label="Input Audio", type="filepath")
                step_stems_project = gr.Textbox(label="Project Name", value="step_project")
                step_stems_btn = gr.Button("Separate Stems")
                step_vocals_output = gr.Audio(label="Vocals")
                step_instrumental_output = gr.Audio(label="Instrumental")
                step_stems_status = gr.Textbox(label="Status")

                def step_separate(audio, project_name, progress=gr.Progress()):
                    project_dir = create_project(project_name)
                    v, i, msg = separate_stems(audio, project_dir, progress)
                    return v, i, msg

                step_stems_btn.click(
                    step_separate,
                    inputs=[step_stems_audio, step_stems_project],
                    outputs=[step_vocals_output, step_instrumental_output, step_stems_status]
                )

            with gr.Accordion("3. Transcribe Lyrics", open=False):
                step_trans_audio = gr.Audio(label="Input Audio (vocals)", type="filepath")
                step_trans_project = gr.Textbox(label="Project Name", value="step_project")
                step_trans_method = gr.Radio(["Whisper", "HeartTranscriptor"], value="Whisper", label="Method")
                step_trans_btn = gr.Button("Transcribe")
                step_trans_output = gr.Textbox(label="SRT Output", lines=10)
                step_trans_status = gr.Textbox(label="Status")

                def step_transcribe(audio, project_name, method, progress=gr.Progress()):
                    project_dir = create_project(project_name)
                    use_hm = method == "HeartTranscriptor"
                    path, text, msg = transcribe_lyrics(audio, project_dir, use_hm, progress)
                    return text, msg

                step_trans_btn.click(
                    step_transcribe,
                    inputs=[step_trans_audio, step_trans_project, step_trans_method],
                    outputs=[step_trans_output, step_trans_status]
                )

        with gr.Tab("Settings"):
            gr.Markdown("### Configuration")

            gr.Markdown(f"""
            **Project Directory:** `{PROJECTS_DIR}`

            **ComfyUI Directory:** `{COMFYUI_DIR}`

            **HeartMuLa Directory:** `{HEARTLIB_DIR}`

            ---

            **ComfyUI Setup:**
            ```bash
            cd {COMFYUI_DIR}
            python main.py --listen 0.0.0.0 --port 8188
            ```

            **Required Wan 2.2 Models:**
            - `wan2.2_s2v_14B_fp8_scaled.safetensors`
            - `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
            - `wav2vec2_large_english_fp16.safetensors`
            - `wan_2.1_vae.safetensors`

            **HeartMuLa Models:**
            - `HeartMuLa-oss-3B`
            - `HeartCodec-oss`
            - `HeartTranscriptor-oss`
            """)

            with gr.Row():
                projects_list = gr.Dropdown(
                    choices=get_project_list(),
                    label="Existing Projects"
                )
                refresh_projects_btn = gr.Button("Refresh")

            def refresh_projects():
                return gr.Dropdown(choices=get_project_list())

            refresh_projects_btn.click(refresh_projects, outputs=projects_list)


if __name__ == "__main__":
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
