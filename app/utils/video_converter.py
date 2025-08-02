import subprocess
import os

def convert_to_browser_compatible(input_path: str, overwrite: bool = True) -> str:
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video file not found: {input_path}")
    
    output_path = input_path if overwrite else input_path.replace(".mp4", "_fixed.mp4")

    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    subprocess.run(command, check=True)

    return output_path
