# Video Chapter Visualizer

A Python tool that adds visual chapter markers to videos, creating an interactive timeline with chapter titles and progress indicators.

## Features

- Adds a sleek chapter timeline to your videos
- Displays current chapter with visual highlighting
- Supports custom chapter titles and timestamps
- Preview mode for quick testing
- Gradient effects and modern UI design

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your chapter file (chapters.json) in the following format:

```json
[
    {
        "title": "Chapter Title",
        "start_time": 0,
        "end_time": 10
    },
    {
        "title": "Next Chapter",
        "start_time": 10,
        "end_time": 20
    }
]
```

2. Run the script with your video file:

```bash
python video_chapter_visualizer.py --video input_video.mp4 --chapters chapters.json --output output_video.mp4
```

### Command Line Arguments

- `--video`, `-v`: Path to the input video file (default: input_video.mp4)
- `--chapters`, `-c`: Path to the JSON chapter file (default: chapters.json)
- `--output`, `-o`: Path for the output video file (default: output_video.mp4)
- `--preview`, `-p`: Preview mode - process only first 10 seconds

### Preview Mode

To quickly test your chapter configuration, use the preview mode:

```bash
python video_chapter_visualizer.py -v input_video.mp4 -c chapters.json -o preview.mp4 --preview
```

## Chapter File Format

The chapters.json file should contain an array of chapter objects, each with:

- `title`: Chapter title (string)
- `start_time`: Start time in seconds (number)
- `end_time`: End time in seconds (number)

Chapters should be arranged in chronological order, and times should not overlap.

## Visual Features

- Gradient background for timeline
- Active chapter highlighting with blue accent
- Dynamic text sizing for different chapter lengths
- Smooth transitions between chapters
- High contrast text for better readability

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Pillow (PIL)

## License

This project is licensed under the MIT License - see the LICENSE file for details.