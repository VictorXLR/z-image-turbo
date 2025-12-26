#!/usr/bin/env python3
"""
Turbo-Term Terminal Viewer
Displays generated images in Ghostty with interactive hotkeys
"""

# Suppress noisy library warnings before imports
import warnings
import os
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCHAO_SILENCE_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import logging

logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Suppress stderr during library imports (torchao prints directly)
_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
try:
    import torch
    from diffusers import ZImagePipeline
finally:
    sys.stderr.close()
    sys.stderr = _stderr

from PIL import Image

import argparse
import base64
import io
import random
import resource
import tempfile
import termios
import time
import tty
import subprocess
from pathlib import Path
from typing import Optional


def getch() -> str:
    """Read a single character without waiting for Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def bell():
    """Ring terminal bell."""
    sys.stdout.write("\a")
    sys.stdout.flush()


def format_bytes(num_bytes: float) -> str:
    """Format bytes as human-readable string."""
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.1f}GB"
    elif num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.0f}MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f}KB"
    return f"{num_bytes:.0f}B"


def memory_bar(used: float, total: float, width: int = 20) -> str:
    """Create a visual memory bar."""
    if total <= 0:
        return "░" * width
    ratio = min(used / total, 1.0)
    filled = int(ratio * width)
    # Color based on usage: green < 50%, yellow < 80%, red >= 80%
    if ratio < 0.5:
        color = "\033[32m"  # Green
    elif ratio < 0.8:
        color = "\033[33m"  # Yellow
    else:
        color = "\033[31m"  # Red
    reset = "\033[0m"
    bar = "█" * filled + "░" * (width - filled)
    return f"{color}{bar}{reset}"


class TerminalImageViewer:
    OUTPUT_DIR = Path.home() / "Pictures" / "Autogen"

    def __init__(self, width: int = 640, height: int = 480, steps: int = 9):
        self.width = width
        self.height = height
        self.steps = steps
        self.pipeline = None
        self.device = None
        self.dtype = None
        self.current_image = None
        self.current_prompt = ""
        self.current_filepath = None
        self.current_seed = None
        self.last_gen_time = 0.0
        self.temp_files = []
        # Ensure output directory exists
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def get_memory_stats(self) -> dict:
        """Get current memory usage stats."""
        stats = {}

        # Process memory (RSS - Resident Set Size)
        # On macOS, ru_maxrss is in bytes
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        stats["process"] = rusage.ru_maxrss  # bytes on macOS

        # GPU memory
        if self.device == "mps" and torch.backends.mps.is_available():
            try:
                stats["gpu_used"] = torch.mps.current_allocated_memory()
                stats["gpu_total"] = torch.mps.driver_allocated_memory()
            except Exception:
                pass
        elif self.device == "cuda" and torch.cuda.is_available():
            try:
                stats["gpu_used"] = torch.cuda.memory_allocated()
                stats["gpu_total"] = torch.cuda.memory_reserved()
            except Exception:
                pass

        return stats

    def format_memory_display(self) -> str:
        """Format memory stats for terminal display."""
        stats = self.get_memory_stats()
        parts = []

        # Process RAM
        if "process" in stats:
            parts.append(f"RAM: {format_bytes(stats['process'])}")

        # GPU memory with bar
        if "gpu_used" in stats and "gpu_total" in stats:
            gpu_used = stats["gpu_used"]
            gpu_total = stats["gpu_total"]
            bar = memory_bar(gpu_used, gpu_total, width=15)
            parts.append(
                f"GPU: {bar} {format_bytes(gpu_used)}/{format_bytes(gpu_total)}"
            )

        return "  ".join(parts) if parts else "Memory: N/A"

    def setup_device(self):
        """Setup device and dtype"""
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.bfloat16
            print("✓ Using Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
            print("✓ Using CUDA")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            print("⚠️  Using CPU (slower)")

    def load_model(self):
        """Load Z-Image-Turbo model"""
        if self.pipeline is not None:
            return

        print("📦 Loading Z-Image-Turbo model...")
        print("   First time will download ~32GB...")

        try:
            self.pipeline = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=self.dtype,
                use_safetensors=True,
            )
            self.pipeline = self.pipeline.to(self.device)

            # Enable memory optimizations
            if self.device in ["mps", "cuda"]:
                try:
                    self.pipeline.enable_attention_slicing()
                except AttributeError:
                    pass  # Method not available in this version

            print("✓ Model loaded successfully!")
            print(f"  {self.format_memory_display()}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def generate_image(self, prompt: str, seed: int = None) -> Optional[Image.Image]:
        """Generate image from prompt"""
        try:
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            self.current_seed = seed
            print(f"\n🎨 Generating: {prompt}")
            print(f"   Seed: {seed}")

            generator = torch.Generator(device=self.device).manual_seed(seed)

            start_time = time.time()
            image = self.pipeline(
                prompt=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
            self.last_gen_time = time.time() - start_time

            self.current_image = image
            self.current_prompt = prompt

            # Auto-save to ~/Pictures/Autogen
            self.current_filepath = self.save_image(image)

            # Show timing
            print(f"⏱️  Generated in {self.last_gen_time:.1f}s")

            # Ring bell when done
            bell()

            return image

        except Exception as e:
            print(f"❌ Error generating image: {e}")
            return None

    def display_image_kitty(self, image: Image.Image, max_width: int = 640) -> bool:
        """Display image using Kitty graphics protocol (works in Ghostty)"""
        try:
            # Resize for terminal display
            ratio = max_width / image.width
            display_image = image.resize(
                (max_width, int(image.height * ratio)), Image.Resampling.LANCZOS
            )

            # Convert to PNG bytes
            buf = io.BytesIO()
            display_image.save(buf, format="PNG")
            image_data = buf.getvalue()

            # Base64 encode
            b64_data = base64.standard_b64encode(image_data).decode("ascii")

            # Write using Kitty graphics protocol with chunking
            chunk_size = 4096
            chunks = [
                b64_data[i : i + chunk_size]
                for i in range(0, len(b64_data), chunk_size)
            ]

            for i, chunk in enumerate(chunks):
                is_last = i == len(chunks) - 1
                if i == 0:
                    # First chunk: include all parameters
                    # a=T: transmit and display, f=100: PNG format, m=0/1: more data flag
                    cmd = f"\033_Ga=T,f=100,m={0 if is_last else 1};{chunk}\033\\"
                else:
                    # Continuation chunks
                    cmd = f"\033_Gm={0 if is_last else 1};{chunk}\033\\"
                sys.stdout.write(cmd)

            sys.stdout.write("\n")
            sys.stdout.flush()
            return True

        except Exception as e:
            print(f"Error displaying image: {e}")
            return False

    def display_image_chafa(self, image_path: str):
        """Fallback: Use chafa for ASCII/ANSI display"""
        try:
            result = subprocess.run(
                ["chafa", image_path, "--size=80x40", "--symbols=block"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(result.stdout)
                return True
        except FileNotFoundError:
            pass
        return False

    def display_image(self, image: Image.Image) -> str:
        """Display image in terminal and return temp file path"""
        # Save to temporary file for fallback/reference
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name
            self.temp_files.append(temp_path)

        # Try Kitty protocol first (Ghostty supports it)
        if not self.display_image_kitty(image):
            # Fallback to chafa if available
            if not self.display_image_chafa(temp_path):
                print(f"\n[Image saved to: {temp_path}]")
                print("(Install 'chafa' for better terminal display)")

        return temp_path

    def save_image(self, image: Image.Image, prompt: str = None) -> Path:
        """Save image to ~/Pictures/Autogen and return filepath"""
        prompt = prompt or self.current_prompt
        # Generate filename from prompt - handle unicode properly
        import re

        safe_prompt = re.sub(r"[^\w\s-]", "", prompt[:30], flags=re.UNICODE)
        safe_prompt = re.sub(r"[-\s]+", "_", safe_prompt).strip("_") or "output"
        counter = 1
        while True:
            filepath = self.OUTPUT_DIR / f"{safe_prompt}_{counter}.png"
            if not filepath.exists():
                break
            counter += 1

        image.save(filepath)
        print(f"✅ Saved: {filepath}")
        return filepath

    def copy_to_clipboard(self, image: Image.Image):
        """Copy image to clipboard (macOS)"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                # Use osascript to copy image to clipboard on macOS
                script = f'''
                    set theFile to POSIX file "{tmp.name}"
                    set theImage to read theFile as «class PNGf»
                    set the clipboard to theImage
                '''
                subprocess.run(["osascript", "-e", script], check=True)
                os.unlink(tmp.name)
            print("✅ Image copied to clipboard")
        except Exception as e:
            print(f"❌ Failed to copy to clipboard: {e}")

    def delete_image(self):
        """Delete the saved image file"""
        if self.current_filepath and self.current_filepath.exists():
            try:
                self.current_filepath.unlink()
                print(f"🗑️  Deleted: {self.current_filepath}")
                self.current_filepath = None
            except Exception as e:
                print(f"❌ Failed to delete: {e}")

    def show_menu(self):
        """Show interactive menu with instant hotkeys"""
        print(f"\n" + "=" * 60)
        print(f"🖼️  Turbo-Term ({self.width}x{self.height}) • {self.last_gen_time:.1f}s")
        print(f"Prompt: {self.current_prompt}")
        print(f"Seed: {self.current_seed}")
        print(f"{self.format_memory_display()}")
        print("=" * 60)
        print("\n  [c] Copy   [d] Delete   [u] Upscale (2x)")
        print("  [v] Variation   [r] Reproduce   [m] Memory")
        print("  [n] New   [q] Quit")
        print("-" * 60)

        while True:
            try:
                sys.stdout.write("\n> ")
                sys.stdout.flush()
                choice = getch().lower()
                print(choice)  # Echo the keypress

                if choice == "c":
                    self.copy_to_clipboard(self.current_image)

                elif choice == "d":
                    self.delete_image()
                    return "new"

                elif choice == "u":
                    # Upscale: same prompt, same seed, 2x resolution
                    return "upscale"

                elif choice == "v":
                    # Variation: same prompt, new random seed
                    return "variation"

                elif choice == "r":
                    # Reproduce: same prompt AND same seed
                    return "reproduce"

                elif choice == "m":
                    print(f"  {self.format_memory_display()}")

                elif choice == "n":
                    return "new"

                elif choice == "q" or choice == "\x03":  # q or Ctrl+C
                    return "quit"

                else:
                    print("Unknown key. Use: c/d/u/v/r/n/m/q")

            except (KeyboardInterrupt, EOFError):
                return "quit"

    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass  # File already deleted or inaccessible

    def run(self):
        """Main interactive loop"""
        self.setup_device()
        self.load_model()

        next_seed = None  # None = random, or specific seed to reproduce

        try:
            while True:
                # Get prompt
                if not self.current_prompt:
                    prompt = input("\nprompt > ").strip()
                    if not prompt:
                        continue
                else:
                    prompt = self.current_prompt

                # Generate image
                image = self.generate_image(prompt, seed=next_seed)
                next_seed = None  # Reset to random for next generation

                if image is None:
                    continue

                # Display image in terminal
                self.display_image(image)

                # Show menu and handle choice
                action = self.show_menu()

                if action == "quit":
                    break
                elif action == "new":
                    self.current_prompt = ""
                    self.current_image = None
                elif action == "variation":
                    # Same prompt, new random seed (next_seed stays None)
                    pass
                elif action == "reproduce":
                    # Same prompt, same seed
                    next_seed = self.current_seed
                elif action == "upscale":
                    # Same prompt, same seed, 2x resolution (max 1024)
                    next_seed = self.current_seed
                    self.width = min(self.width * 2, 1024)
                    self.height = min(self.height * 2, 1024)
                    print(f"📐 Upscaling to {self.width}x{self.height}...")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Turbo-Term: Generate images in your terminal with Z-Image-Turbo"
    )
    parser.add_argument(
        "-W", "--width", type=int, default=640, help="Image width (default: 640)"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=480, help="Image height (default: 480)"
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=9, help="Inference steps (default: 9)"
    )
    args = parser.parse_args()

    viewer = TerminalImageViewer(width=args.width, height=args.height, steps=args.steps)
    viewer.run()


if __name__ == "__main__":
    main()
