import subprocess
import os
import tempfile
import shutil

def convert_to_browser_compatible(input_path: str, overwrite: bool = True) -> str:
    """
    Convert video to browser-compatible format using FFmpeg.
    
    Args:
        input_path: Path to the input video file
        overwrite: If True, replace the original file. If False, create a new file with "_fixed" suffix
    
    Returns:
        Path to the converted video file
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        subprocess.CalledProcessError: If FFmpeg conversion fails
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video file not found: {input_path}")
    
    if overwrite:
        # Create a temporary file to avoid input/output conflict
        temp_dir = os.path.dirname(input_path)
        with tempfile.NamedTemporaryFile(suffix='.mp4', dir=temp_dir, delete=False) as temp_file:
            temp_output_path = temp_file.name
        
        try:
            command = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                temp_output_path,
            ]
            
            # Run FFmpeg with error handling
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                # Clean up temp file on error
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
                raise subprocess.CalledProcessError(
                    result.returncode, 
                    command, 
                    output=result.stdout, 
                    stderr=result.stderr
                )
            
            # Replace original file with converted version
            shutil.move(temp_output_path, input_path)
            return input_path
            
        except Exception as e:
            # Clean up temp file on any error
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            raise e
    
    else:
        # Create new file with "_fixed" suffix
        base_name, ext = os.path.splitext(input_path)
        output_path = f"{base_name}_fixed{ext}"
        
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        
        # Run FFmpeg with error handling
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, 
                command, 
                output=result.stdout, 
                stderr=result.stderr
            )
        
        return output_path