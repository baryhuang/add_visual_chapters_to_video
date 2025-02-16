import cv2
import json
import numpy as np
import time
from typing import Dict, List, Tuple
from font_manager import FontManager
from PIL import Image, ImageDraw, ImageFont

output_done_in_second = {}

class VideoChapterVisualizer:
    def __init__(self, video_path: str, chapter_file_path: str):
        """Initialize the video chapter visualizer.

        Args:
            video_path (str): Path to the video file
            chapter_file_path (str): Path to the JSON chapter file
        """
        self.video_path = video_path
        self.chapter_file_path = chapter_file_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.chapters = self._load_chapters()
        self.font_manager = FontManager()

    def _load_chapters(self) -> List[Dict]:
        """Load chapters from JSON file.

        Returns:
            List[Dict]: List of chapter dictionaries with start_time, end_time, and title
        """
        with open(self.chapter_file_path, 'r') as f:
            chapters = json.load(f)
        return chapters

    def _draw_chapter_markers(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Draw chapter markers on the frame.

        Args:
            frame (np.ndarray): Current video frame
            current_time (float): Current video time in seconds

        Returns:
            np.ndarray: Frame with chapter markers
        """
        # Calculate dimensions for chapter markers
        marker_height = int(self.frame_height * 0.06)  # 6% of video height
        marker_y = self.frame_height - marker_height - 15  # 15px padding from bottom
        total_width = self.frame_width - 30  # 15px padding on each side
        marker_x = 15

        # Create gradient background
        gradient = np.linspace(0.2, 0.4, total_width).reshape(1, -1)
        gradient = np.tile(gradient, (marker_height, 1))
        bg = np.zeros((marker_height, total_width, 3), dtype=np.uint8)
        bg[:, :, 0] = bg[:, :, 1] = bg[:, :, 2] = (gradient * 255).astype(np.uint8)

        # Apply background with alpha blending
        roi = frame[marker_y:marker_y+marker_height, marker_x:marker_x+total_width]
        frame[marker_y:marker_y+marker_height, marker_x:marker_x+total_width] = \
            cv2.addWeighted(roi, 0.3, bg, 0.7, 0)

        # Draw chapter segments
        video_duration = self.total_frames / self.fps
        for chapter in self.chapters:
            start_time = chapter['start_time']
            end_time = chapter['end_time']
            title = chapter['title']

            # Calculate position and width
            start_x = marker_x + int((start_time / video_duration) * total_width)
            end_x = marker_x + int((end_time / video_duration) * total_width)
            width = min(max(end_x - start_x, 1), total_width)  # Ensure width is within bounds

            # Determine if chapter is active
            is_active = start_time <= current_time <= end_time
            
            # Create solid color for segments
            if is_active:
                segment_color = (211, 211, 211)  # Light gray (#D3D3D3)
            else:
                segment_color = (64, 64, 64)    # Dark gray (#404040)

            # Create solid color segment
            segment = np.zeros((marker_height, width, 3), dtype=np.uint8)
            segment[:, :] = segment_color

            # Apply segment with alpha blending
            roi = frame[marker_y:marker_y+marker_height, start_x:end_x]
            # Skip segments with invalid dimensions
            if width <= 0 or roi.shape[1] <= 0:
                continue
            # Ensure segment matches ROI dimensions
            if roi.shape != segment.shape:
                segment = cv2.resize(segment, (roi.shape[1], roi.shape[0]))
            frame[marker_y:marker_y+marker_height, start_x:end_x] = \
                cv2.addWeighted(roi, 0.1, segment, 0.9, 0)

            # Add subtle border effect
            border_color = (128, 128, 128) if is_active else (64, 64, 64)  # #808080 or #404040
            cv2.rectangle(frame,
                         (start_x, marker_y),
                         (end_x, marker_y + marker_height),
                         border_color,
                         1)

            # Add blue highlight bar for active chapter
            if is_active:
                highlight_height = 3  # Height of the blue bar
                highlight_y = marker_y - highlight_height - 2  # Position above the chapter marker
                # Create gradient blue highlight
                highlight = np.zeros((highlight_height, width, 3), dtype=np.uint8)
                for i in range(width):
                    ratio = i / width
                    # Gradient from darker to lighter blue
                    blue_start = (41, 98, 255)  # Darker blue
                    blue_end = (66, 135, 245)   # Lighter blue
                    color = tuple(int(c1 + (c2 - c1) * ratio) for c1, c2 in zip(blue_start, blue_end))
                    highlight[:, i] = color
                
                # Apply highlight with alpha blending
                roi = frame[highlight_y:highlight_y+highlight_height, start_x:end_x]
                if width > 0 and roi.shape[1] > 0:
                    if roi.shape != highlight.shape:
                        highlight = cv2.resize(highlight, (roi.shape[1], roi.shape[0]))
                    frame[highlight_y:highlight_y+highlight_height, start_x:end_x] = \
                        cv2.addWeighted(roi, 0.2, highlight, 0.8, 0)

            # Calculate text position for perfect centering
            font_size = 24 if is_active else 20  # Increased font size for better visibility
            img_pil = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(self.font_manager.get_font_path(), font_size)
            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Ensure text width doesn't exceed segment width
            if text_width > width * 0.9:  # Allow text to take up to 90% of segment width
                # Scale down font size to fit
                scale_factor = (width * 0.9) / text_width
                font_size = int(font_size * scale_factor)
                font = ImageFont.truetype(self.font_manager.get_font_path(), font_size)
                text_bbox = draw.textbbox((0, 0), title, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

            # Only draw text if segment is wide enough
            if width > text_width + 20:  # Increased padding for better spacing
                # Ensure text stays within frame boundaries
                text_x = max(marker_x, min(start_x + (width - text_width) // 2, marker_x + total_width - text_width))
                text_y = marker_y + (marker_height - text_height) // 2
                
                # Draw text with modern styling - no shadows for cleaner look
                text_color = (64, 64, 64) if is_active else (128, 128, 128)  # Dark gray (#404040) or #808080
                frame = self.font_manager.put_text_pil(frame, title,
                                                      (text_x, text_y),
                                                      font_size=font_size,
                                                      color=text_color)
                
                # Debug information - only output if current_time is a multiple of 2 seconds and this is the active chapter
                if int(current_time) % 2 == 0 and is_active:
                    if int(current_time) not in output_done_in_second:
                        output_done_in_second[int(current_time)] = True
                        print(f"\nTime: {current_time:.1f} seconds")
                        print(f"Video Frame Dimensions: {self.frame_width}x{self.frame_height}")
                        print(f"Chapter Block: x={marker_x}, y={marker_y}, width={total_width}, height={marker_height}")
                        print(f"Chapter: {title}")
                        print(f"Text Position: ({text_x}, {text_y})")
                        print(f"Text Dimensions: {text_width}x{text_height}")
                        print(f"Segment Width: {width}\n")

        return frame

    def process_video(self, output_path: str):
        """Process the video and add chapter visualizations.

        Args:
            output_path (str): Path where the processed video will be saved
        """
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps,
                            (self.frame_width, self.frame_height))

        frame_count = 0
        total_frames = self.total_frames
        start_time = time.time()

        print(f"Starting video processing...\nTotal frames: {total_frames}")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = frame_count / self.fps
            # In preview mode, stop after 10 seconds
            if hasattr(self, 'preview_mode') and self.preview_mode and current_time >= 10:
                break

            frame_with_chapters = self._draw_chapter_markers(frame, current_time)
            out.write(frame_with_chapters)

            # Update progress every 2 seconds
            frame_count += 1
            if int(current_time) % 2 == 0 and frame_count % int(self.fps) == 0:
                elapsed_time = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                frames_per_second = frame_count / elapsed_time
                estimated_time = (total_frames - frame_count) / frames_per_second

                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                print(f"Processing speed: {frames_per_second:.1f} fps")
                print(f"Estimated time remaining: {estimated_time:.1f} seconds\n")

        # Release resources
        self.cap.release()
        out.release()

        print(f"\nProcessing complete!\nOutput saved to: {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add chapter visualizations to a video')
    parser.add_argument('--video', '-v', type=str, default='input_video.mp4',
                        help='Path to the input video file')
    parser.add_argument('--chapters', '-c', type=str, default='chapters.json',
                        help='Path to the JSON chapter file')
    parser.add_argument('--output', '-o', type=str, default='output_video.mp4',
                        help='Path for the output video file')
    parser.add_argument('--preview', '-p', action='store_true',
                        help='Preview mode: process only first 10 seconds')

    args = parser.parse_args()

    visualizer = VideoChapterVisualizer(args.video, args.chapters)
    if args.preview:
        visualizer.preview_mode = True
        # Calculate frames for 10 seconds
        visualizer.total_frames = min(int(visualizer.fps * 10), visualizer.total_frames)
    visualizer.process_video(args.output)

if __name__ == '__main__':
    main()