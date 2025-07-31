import subprocess
import os

def convert_to_browser_compatible(input_path: str) -> str:
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    
    temp_path = os.path.join(dir_name, f"{name}_temp{ext}")
    
    command = [
        r"C:/Users/yasir/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe",
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-strict', 'experimental',
        temp_path
    ]

    print("Running command:", " ".join(command))
    
    try:
        completed_process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("FFmpeg output:", completed_process.stdout)
        print("FFmpeg errors (if any):", completed_process.stderr)
        
        # Replace the original file with the converted temp file
        os.replace(temp_path, input_path)
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:", e.stderr)
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except Exception as e:
        # Clean up temp file if it exists to avoid clutter
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Post-processing failed: {e}")

    print(f"Replaced original video with browser-compatible video: {input_path}")
    return input_path
