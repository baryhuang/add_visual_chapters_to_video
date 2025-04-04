#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create chapters.json file from video transcription using LLM processing.
This script analyzes video content and generates meaningful chapter markers.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import assemblyai as aai
from anthropic import Anthropic
import openai
import time
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import jieba
import zhconv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing information."""
    text: str
    start: float
    end: float
    words: List[Dict[str, Any]]

# Load environment variables
load_dotenv()

# Get API keys from environment
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not ASSEMBLY_API_KEY:
    logger.error("AssemblyAI API key not found. Please set the ASSEMBLY_API_KEY environment variable.")
    raise ValueError("AssemblyAI API key not found")

if not ANTHROPIC_API_KEY:
    logger.error("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
    raise ValueError("Anthropic API key not found")

if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    raise ValueError("OpenAI API key not found")

# Initialize API clients
aai.settings.api_key = ASSEMBLY_API_KEY
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.Client(api_key=OPENAI_API_KEY)

def create_chapters_from_transcription(transcription_segments: List[Dict[str, Any]], video_duration: float, detailed_transcription: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Use LLM to analyze transcription and create meaningful chapter markers.
    
    Args:
        transcription_segments: List of transcription segments with timing information
        video_duration: Total duration of the video in seconds
        detailed_transcription: Dictionary containing detailed transcription data with word-level timestamps
    
    Returns:
        List of chapter dictionaries with title, start_time, and end_time
    """
    # Format transcription for LLM with word-level timestamps if available
    if detailed_transcription and 'utterances' in detailed_transcription:
        transcript_lines = []
        for i, utterance in enumerate(detailed_transcription['utterances']):
            # Add utterance header
            transcript_lines.append(f"\n=== Utterance {i+1} ({utterance['start']:.2f}s - {utterance['end']:.2f}s) ===")
            
            # Add word-level details
            word_lines = []
            for word in utterance['words']:
                word_lines.append(f"{word['text']}[{word['start']:.2f}-{word['end']:.2f}]")
            transcript_lines.append(" ".join(word_lines))
        
        transcript_text = "\n".join(transcript_lines)
    else:
        # Fallback to basic transcription format
        transcript_text = "\n".join([
            f"{i+1}. {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}"
            for i, seg in enumerate(transcription_segments)
        ])
    
    # First use OpenAI to identify potential chapter breaks
    logger.info("Using OpenAI to identify chapter breaks...")
    
    openai_prompt = f"""你是一位專業的影片章節編輯。請分析以下逐字稿，並創建有意義的章節標記。
逐字稿包含了每個字的精確時間戳記，格式為：文字[開始時間-結束時間]

要求：
1. 每個章節長度應該在3-5分鐘之間
2. 章節之間要有自然的過渡，最好在句子或段落的自然結束點
3. 章節標題要簡潔有力，突出重點
4. 章節標題要符合聖經教義和教會用語
5. 整個視頻總長度是 {video_duration:.2f} 秒
6. 使用字詞級別的時間戳記來確保章節分割點的精確性

逐字稿：
{transcript_text}

請用以下格式輸出章節建議：
```json
{{
  "chapters": [
    {{
      "title": "章節標題",
      "start_time": 開始時間（秒）,
      "end_time": 結束時間（秒）,
      "reason": "為什麼在這裡分章節的原因，包括提到具體的關鍵字詞和時間點"
    }}
  ]
}}
```"""

    try:
        openai_response = openai_client.chat.completions.create(
            model="o3-mini",
            messages=[
                {
                    "role": "system",
                    "content": "你是一位專業的影片章節編輯，專門為教會視頻創建章節標記。"
                },
                {
                    "role": "user",
                    "content": openai_prompt
                }
            ]
        )
        
        # Extract JSON from OpenAI response
        response_text = openai_response.choices[0].message.content
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            openai_data = json.loads(json_str)
            
            # Verify and refine chapters with Claude
            logger.info("Using Claude to verify and refine chapter titles...")
            
            claude_prompt = f"""請審查並優化以下視頻章節建議。確保章節標題符合聖經教義，並且清晰準確地反映內容。

原始章節建議：
{json.dumps(openai_data, ensure_ascii=False, indent=2)}

請考慮：
1. 標題是否符合聖經教義和教會用語
2. 時間點的分配是否合理
3. 章節之間的過渡是否自然
4. 標題是否簡潔有力

請用相同的 JSON 格式提供優化後的章節建議。如果原始建議已經很好，可以保持不變。"""

            claude_response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.2,
                system="你是一位資深的教會影片編輯，專門處理講道和敬拜視頻的章節劃分。",
                messages=[{"role": "user", "content": claude_prompt}]
            )
            
            # Extract JSON from Claude response
            response_text = claude_response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                claude_data = json.loads(json_str)
                
                # Format chapters for output
                chapters = []
                for chapter in claude_data['chapters']:
                    chapters.append({
                        'title': chapter['title'],
                        'start_time': float(chapter['start_time']),
                        'end_time': float(chapter['end_time'])
                    })
                
                return chapters
            
            else:
                logger.error("Failed to extract JSON from Claude response")
                return None
        
        else:
            logger.error("Failed to extract JSON from OpenAI response")
            return None
    
    except Exception as e:
        logger.error(f"Error creating chapters: {e}")
        return None

def save_chapters(chapters: List[Dict[str, Any]], output_path: str) -> bool:
    """
    Save chapters to a JSON file.
    
    Args:
        chapters: List of chapter dictionaries
        output_path: Path to save the chapters.json file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save chapters to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chapters, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Saved chapters to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving chapters: {e}")
        return False

def load_chapters(chapters_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load chapters from a JSON file.
    
    Args:
        chapters_path: Path to the chapters.json file
    
    Returns:
        List of chapter dictionaries if successful, None otherwise
    """
    try:
        with open(chapters_path, 'r', encoding='utf-8') as f:
            chapters = json.load(f)
        logger.info(f"Loaded chapters from {chapters_path}")
        return chapters
    except Exception as e:
        logger.error(f"Error loading chapters: {e}")
        return None

def display_chapters(chapters: List[Dict[str, Any]]) -> None:
    """
    Display chapters in a readable format.
    
    Args:
        chapters: List of chapter dictionaries
    """
    print("\nVideo Chapters:")
    print("-" * 50)
    for i, chapter in enumerate(chapters, 1):
        duration = chapter['end_time'] - chapter['start_time']
        start_min = int(chapter['start_time'] // 60)
        start_sec = int(chapter['start_time'] % 60)
        end_min = int(chapter['end_time'] // 60)
        end_sec = int(chapter['end_time'] % 60)
        dur_min = int(duration // 60)
        dur_sec = int(duration % 60)
        
        print(f"\nChapter {i}:")
        print(f"Title: {chapter['title']}")
        print(f"Time: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d} (Duration: {dur_min:02d}:{dur_sec:02d})")

def get_video_duration_from_transcription(transcription_segments: List[Dict[str, Any]], default_duration: float = 900.0) -> float:
    """
    Calculate video duration from transcription segments.
    
    Args:
        transcription_segments: List of transcription segments with timing information
        default_duration: Default duration in seconds (default: 900.0, which is 15 minutes)
    
    Returns:
        float: Total duration in seconds, defaults to 15 minutes if no valid duration found
    """
    if not transcription_segments:
        logger.info(f"No transcription segments found, using default duration of {default_duration:.2f} seconds")
        return default_duration
    
    try:
        # Find the latest end time from all segments
        max_end_time = max(float(seg['end']) for seg in transcription_segments)
        # If the duration seems too short (less than 1 minute), use default
        if max_end_time < 60:
            logger.warning(f"Calculated duration ({max_end_time:.2f}s) seems too short, using default duration of {default_duration:.2f} seconds")
            return default_duration
        return max_end_time
    except (ValueError, KeyError) as e:
        logger.warning(f"Error calculating duration from segments: {e}. Using default duration of {default_duration:.2f} seconds")
        return default_duration

def save_transcription(transcription_data: Dict[str, Any], output_path: str) -> bool:
    """
    Save transcription data to a JSON file.
    
    Args:
        transcription_data: Dictionary containing transcription information
        output_path: Path to save the transcription JSON file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save transcription to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Saved transcription to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving transcription: {e}")
        return False

def transcribe_video(video_path: str, language_code: str = "zh") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Transcribe the video using AssemblyAI and return segments with timing information.
    First extracts audio from video, then transcribes the audio.
    Uses word-level timestamps for more precise timing information.
    If a transcript file already exists, load it instead of re-transcribing.
    
    Args:
        video_path: Path to the video file
        language_code: Language code (zh for Traditional Chinese)
    
    Returns:
        Tuple containing:
        - List of transcription segments with timing information
        - Dictionary containing detailed transcription data
    """
    # Create transcript file path
    video_path_obj = Path(video_path)
    transcript_path = video_path_obj.with_suffix('.transcript.json')
    
    # Check if transcript file exists
    if transcript_path.exists():
        logger.info(f"Loading existing transcript from {transcript_path}")
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            return process_transcript_data(transcript_data, video_path, language_code)
        except Exception as e:
            logger.warning(f"Failed to load existing transcript: {e}. Will re-transcribe.")
    
    logger.info(f"Extracting audio from video: {video_path}")
    
    # Extract audio to temporary file
    temp_audio_path = video_path_obj.with_suffix('.temp.wav')
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
        video.close()
        
        logger.info(f"Transcribing audio...")
        
        # Create the transcriber with configuration
        config = aai.TranscriptionConfig(
            language_code=language_code,
            speaker_labels=True,
            punctuate=True,
            format_text=True,
        )
        transcriber = aai.Transcriber()
        
        # Start the transcription with audio file
        transcript = transcriber.transcribe(str(temp_audio_path), config=config)
        
        # Show progress bar while waiting for transcription
        with tqdm(total=100, desc="Transcribing") as pbar:
            last_progress = 0
            while True:
                status = transcript.status
                
                # Calculate progress percentage based on status
                if status == "queued":
                    progress = 5
                elif status == "processing":
                    progress = 50
                elif status == "completed":
                    progress = 100
                    break
                elif status == "error":
                    error_msg = getattr(transcript, 'error', 'Unknown error')
                    raise Exception(f"Transcription failed with status: {status}, error: {error_msg}")
                
                # Update progress bar
                if progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress
                
                time.sleep(3)  # Wait 3 seconds before checking again
                transcript = transcriber.get_transcript(transcript.id)
        
        # Save the complete transcript response as JSON
        # Convert transcript object to dictionary safely
        transcript_dict = {}
        
        # Extract basic properties
        for attr in ['id', 'status', 'text', 'confidence', 'audio_url', 'audio_duration']:
            if hasattr(transcript, attr):
                transcript_dict[attr] = getattr(transcript, attr)
        
        # Extract words
        if hasattr(transcript, 'words') and transcript.words:
            transcript_dict['words'] = []
            for word in transcript.words:
                word_dict = {}
                for word_attr in ['text', 'start', 'end', 'confidence', 'speaker']:
                    if hasattr(word, word_attr):
                        word_dict[word_attr] = getattr(word, word_attr)
                transcript_dict['words'].append(word_dict)
        
        # Extract utterances if available
        if hasattr(transcript, 'utterances') and transcript.utterances:
            transcript_dict['utterances'] = []
            for utterance in transcript.utterances:
                utterance_dict = {}
                for utt_attr in ['text', 'start', 'end', 'confidence', 'speaker']:
                    if hasattr(utterance, utt_attr):
                        utterance_dict[utt_attr] = getattr(utterance, utt_attr)
                
                # Extract words in utterance if available
                if hasattr(utterance, 'words') and utterance.words:
                    utterance_dict['words'] = []
                    for word in utterance.words:
                        word_dict = {}
                        for word_attr in ['text', 'start', 'end', 'confidence']:
                            if hasattr(word, word_attr):
                                word_dict[word_attr] = getattr(word, word_attr)
                        utterance_dict['words'].append(word_dict)
                
                transcript_dict['utterances'].append(utterance_dict)
        
        # Save transcript to file
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Transcription complete and saved to {transcript_path}")
        return process_transcript_data(transcript_dict, video_path, language_code)
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise
    
    finally:
        # Clean up temporary audio file
        if temp_audio_path.exists():
            temp_audio_path.unlink()
            logger.info("Cleaned up temporary audio file")

def process_transcript_data(transcript_data: Dict[str, Any], video_path: str, language_code: str = "zh") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process the transcript data from AssemblyAI response.
    
    Args:
        transcript_data: Complete transcript response from AssemblyAI
        video_path: Path to the video file
        language_code: Language code (zh for Traditional Chinese)
    
    Returns:
        Tuple containing:
        - List of transcription segments with timing information
        - Dictionary containing detailed transcription data
    """
    # Extract utterances and words
    utterances = transcript_data.get('utterances', [])
    words = transcript_data.get('words', [])
    
    # Create segments from utterances
    segments = []
    
    # If we have utterances, use them
    if utterances:
        for utterance in utterances:
            # Convert timestamps from milliseconds to seconds if needed
            start_time = utterance.get('start', 0)
            end_time = utterance.get('end', 0)
            
            # Handle different timestamp formats
            if isinstance(start_time, (int, float)) and start_time > 1000:
                start_time = start_time / 1000
            if isinstance(end_time, (int, float)) and end_time > 1000:
                end_time = end_time / 1000
            
            # Convert text to traditional Chinese if needed
            text = utterance.get('text', '')
            if language_code == "zh":
                text = zhconv.convert(text, 'zh-hant')
                
            segments.append({
                'text': text,
                'start': start_time,
                'end': end_time,
                'speaker': utterance.get('speaker', 'Speaker 1'),
                'confidence': utterance.get('confidence', 1.0),
                'words': utterance.get('words', [])
            })
    
    # If no utterances but we have words, create segments from words
    elif words:
        current_segment = []
        current_words = []
        current_start = None
        
        for word in words:
            # Get word start time, converting from ms to seconds if needed
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)
            
            # Handle different timestamp formats
            if isinstance(word_start, (int, float)) and word_start > 1000:
                word_start = word_start / 1000
            if isinstance(word_end, (int, float)) and word_end > 1000:
                word_end = word_end / 1000
                
            if not current_start:
                current_start = word_start
            
            # Get word text
            word_text = word.get('text', '')
            
            word_info = {
                'text': word_text,
                'start': word_start,
                'end': word_end,
                'confidence': word.get('confidence', 1.0)
            }
            
            current_segment.append(word_text)
            current_words.append(word_info)
            
            # Create a new segment when we hit any of these conditions:
            # 1. Natural sentence ending punctuation
            # 2. Long pause (> 1.5 seconds)
            # 3. Sentence is getting too long (> 50 characters)
            # 4. Natural break point detected by jieba
            should_break = False
            
            # Check for sentence ending punctuation
            if any(p in word_text for p in '.!?。！？'):
                should_break = True
            
            # Check for long pause (if not the first word)
            elif len(current_words) > 1:
                pause_duration = word_start - current_words[-2].get('end', 0)
                if pause_duration > 1.5:  # 1.5 second pause threshold
                    should_break = True
            
            # Check sentence length
            elif len(' '.join(current_segment)) > 50:
                # Use jieba to find a good break point
                words_list = list(jieba.cut(' '.join(current_segment)))
                if len(words_list) > 1:  # If we can actually split it
                    should_break = True
            
            # Check for natural break using jieba
            elif len(' '.join(current_segment)) > 15:  # Only check if we have enough text
                words_list = list(jieba.cut(' '.join(current_segment)))
                # Break if we detect a complete phrase or clause
                if any(w in ['的', '了', '和', '與', '但是', '所以', '因為', '如果'] for w in words_list[-2:]):
                    should_break = True
            
            if should_break:
                text = ' '.join(current_segment)
                # Convert to traditional Chinese if needed
                if language_code == "zh":
                    text = zhconv.convert(text, 'zh-hant')
                
                segments.append({
                    'text': text,
                    'start': current_start,
                    'end': word_end,
                    'speaker': word.get('speaker', 'Speaker 1'),
                    'confidence': sum(w.get('confidence', 1.0) for w in current_words) / max(len(current_words), 1),
                    'words': current_words.copy()
                })
                current_segment = []
                current_words = []
                current_start = None
        
        # Add any remaining words as a segment
        if current_segment:
            text = ' '.join(current_segment)
            # Convert to traditional Chinese if needed
            if language_code == "zh":
                text = zhconv.convert(text, 'zh-hant')
            
            segments.append({
                'text': text,
                'start': current_start,
                'end': current_words[-1].get('end', 0) if current_words else 0,
                'speaker': 'Speaker 1',
                'confidence': sum(w.get('confidence', 1.0) for w in current_words) / max(len(current_words), 1),
                'words': current_words
            })
    
    # If we have neither utterances nor words but have text, create a single segment
    elif 'text' in transcript_data:
        text = transcript_data.get('text', '')
        # Convert to traditional Chinese if needed
        if language_code == "zh":
            text = zhconv.convert(text, 'zh-hant')
            
        segments.append({
            'text': text,
            'start': 0,
            'end': transcript_data.get('audio_duration', 0) / 1000 if isinstance(transcript_data.get('audio_duration'), (int, float)) else 0,
            'speaker': 'Speaker 1',
            'confidence': transcript_data.get('confidence', 1.0),
            'words': []
        })
    
    # Create detailed transcription with metadata
    audio_duration = transcript_data.get('audio_duration', 0)
    if isinstance(audio_duration, (int, float)) and audio_duration > 1000:
        audio_duration = audio_duration / 1000
        
    detailed_transcription = {
        'metadata': {
            'video_path': video_path,
            'duration': audio_duration,
            'language': language_code,
            'language_confidence': transcript_data.get('language_confidence', 1.0),
            'created_at': transcript_data.get('created', str(time.time())),
            'transcript_id': transcript_data.get('id', '')
        },
        'utterances': segments
    }
    
    # Add additional metadata if available
    for key in ['auto_highlights_result', 'content_safety_labels', 'iab_categories_result', 
                'entities', 'summary', 'chapters']:
        if key in transcript_data:
            detailed_transcription['metadata'][key] = transcript_data[key]
    
    logger.info(f"Processed {len(segments)} segments from transcript")
    return segments, detailed_transcription

def main():
    """Main function to create chapters from transcription"""
    import argparse
    
    DEFAULT_DURATION = 900.0  # 15 minutes in seconds
    
    parser = argparse.ArgumentParser(description='Create chapters.json from video transcription')
    parser.add_argument('--video', '-v', type=str, required=True,
                      help='Path to the video file')
    parser.add_argument('--output', '-o', type=str,
                      help='Path to save the chapters.json file (default: <video_folder>/<video_name>_chapters.json)')
    parser.add_argument('--transcription-output', '-t', type=str,
                      help='Path to save the detailed transcription JSON file (default: <video_folder>/<video_name>_transcription.json)')
    parser.add_argument('--video-duration', '-d', type=float,
                      help=f'Total duration of the video in seconds (optional, defaults to {DEFAULT_DURATION} seconds if not provided or cannot be calculated)')
    
    args = parser.parse_args()
    
    try:
        # Get video directory and name
        video_dir = os.path.dirname(os.path.abspath(args.video))
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        
        # Set default output paths if not provided
        if not args.output:
            args.output = os.path.join(video_dir, f"{video_name}_chapters.json")
        if not args.transcription_output:
            args.transcription_output = os.path.join(video_dir, f"{video_name}_transcription.json")
        
        logger.info(f"Video file: {args.video}")
        logger.info(f"Chapters will be saved to: {args.output}")
        logger.info(f"Transcription will be saved to: {args.transcription_output}")
        
        # Transcribe video
        transcription_segments, detailed_transcription = transcribe_video(args.video)
        if not transcription_segments or not detailed_transcription:
            logger.error("Failed to transcribe video")
            return
        
        # Save detailed transcription
        if not save_transcription(detailed_transcription, args.transcription_output):
            logger.error("Failed to save transcription")
            return
        
        # Get video duration from transcription metadata
        video_duration = args.video_duration
        if video_duration is None:
            video_duration = detailed_transcription['metadata']['duration']
            if video_duration <= 0:
                video_duration = get_video_duration_from_transcription(transcription_segments, DEFAULT_DURATION)
            logger.info(f"Using video duration: {video_duration:.2f} seconds")
        
        if video_duration <= 0:
            logger.warning(f"Invalid video duration, using default duration of {DEFAULT_DURATION:.2f} seconds")
            video_duration = DEFAULT_DURATION
        
        # Create chapters using detailed transcription
        chapters = create_chapters_from_transcription(transcription_segments, video_duration, detailed_transcription)
        if not chapters:
            logger.error("Failed to create chapters")
            return
        
        # Display chapters for review
        display_chapters(chapters)
        
        while True:
            choice = input("\nDo you want to:\n1. Save these chapters\n2. Regenerate chapters\n3. Exit\nEnter your choice (1/2/3): ").strip()
            if choice == '1':
                if save_chapters(chapters, args.output):
                    print(f"\nChapters saved to {args.output}")
                break
            elif choice == '2':
                print("\nRegenerating chapters...")
                chapters = create_chapters_from_transcription(transcription_segments, video_duration, detailed_transcription)
                if chapters:
                    display_chapters(chapters)
                    continue
                else:
                    print("Failed to regenerate chapters")
                    break
            elif choice == '3':
                print("Exiting without saving")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 