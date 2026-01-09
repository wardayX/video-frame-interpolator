import argparse
import os
import subprocess
from src.inference import process_video

def main():
    parser = argparse.ArgumentParser(description="Video Interpolator CLI")
    parser.add_argument('--input', type=str, required=True, help="Input video path")
    parser.add_argument('--output', type=str, default="output.mp4", help="Output video path")
    parser.add_argument('--multi', type=int, default=2, help="Frame multiplier (2, 4, 8)")
    parser.add_argument('--crf', type=int, default=17, help="CRF quality for compression")
    
    args = parser.parse_args()
    
    # 1. Run Inference
    temp_output = "temp_interpolated.mp4"
    print(f"Processing {args.input} with {args.multi}x interpolation...")
    process_video(args.input, temp_output, args.multi)
    
    # 2. Re-encode with FFmpeg (requires ffmpeg installed on system)
    print("Re-encoding video...")
    cmd = [
        "ffmpeg", "-i", temp_output, 
        "-c:v", "libx264", 
        "-crf", str(args.crf), 
        "-preset", "fast", 
        "-y", args.output,
        "-loglevel", "error"
    ]
    subprocess.run(cmd)
    
    if os.path.exists(temp_output):
        os.remove(temp_output)
        
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()