import cv2
import os
import numpy as np
import time
import argparse
from typing import List, Tuple, Dict
from datetime import datetime
import json
import base64

def process_with_gpt4o(messages: List[Dict], json_schema: Dict) -> Dict:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        print(f"[DEBUG] process_with_gpt4o - Calling OpenAI API with {len(messages)} messages")
        print(f"[DEBUG] process_with_gpt4o - First message content type: {type(messages[0]['content'])}")
        print(f"[DEBUG] process_with_gpt4o - Function name: {json_schema['name']}")
        
        start_time = datetime.now()
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=[json_schema],
            function_call={"name": json_schema["name"]}  # Force function call
        )
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Check if function call exists in response
        if not response.choices[0].message.function_call:
            # Return empty result with proper structure
            return {
                "contacts": []
            }
            
        try:
            result = json.loads(response.choices[0].message.function_call.arguments)
        except json.JSONDecodeError as e:
            raise

        print(f"[DEBUG] process_with_gpt4o - Returning result with {len(result.get('contacts', []))} contacts in {duration:.2f} seconds")
        return result
    except Exception as e:
        import traceback
        print(f"Error processing with GPT API: {str(e)}")
        print(traceback.format_exc())
        return {"contacts": []}

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return ""

def extract_contact_info(search_query: str, prompt: str, image_path: str) -> Dict:
    """
    Extract contact information from an image using GPT-4o vision capabilities.
    
    Args:
        search_query: Short description of the search
        prompt: Prompt text to send to the API
        image_path: Path to the image file to analyze
        
    Returns:
        Dict: Extracted contact information
    """
    # Verify the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return {"contacts": []}
    
    # Encode the image to base64
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return {"contacts": []}
    
    # Create message with both text and image content
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Extract contact information from this image. Include name, title, company, email, phone, and website."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }]

    json_schema = {
        'name': 'extract_contact_info',
        'description': f'Extract contact information from the image',
        'parameters': {
            'type': 'object',
            'properties': {
                'contacts': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string'},
                            'title': {'type': 'string'},
                            'company': {'type': 'string'},
                            'email': {'type': 'string'},
                            'phone': {'type': 'string'},
                            'website': {'type': 'string'}
                        },
                        'required': ['name', 'title', 'company', 'email', 'phone', 'website']
                    }
                }
            },
            'required': ['contacts']
        }
    }

    return process_with_gpt4o(messages, json_schema)

def save_contacts_to_json(output_dir: str, contacts: List[Dict], frame_info: Dict) -> None:
    """
    Save contacts to a separate JSON file with deduplication based on name and company.
    
    Args:
        output_dir: Directory to save the contacts JSON file
        contacts: List of contact dictionaries to save
        frame_info: Information about the frame where these contacts were found
    """
    contacts_json_path = os.path.join(output_dir, "contacts.json")
    existing_contacts = {}
    
    # Load existing contacts if the file exists
    if os.path.exists(contacts_json_path):
        try:
            with open(contacts_json_path, 'r') as f:
                data = json.load(f)
                # Create a dictionary for faster lookup using name+company as key
                for contact in data.get("contacts", []):
                    key = f"{contact.get('name', '').lower()}_{contact.get('company', '').lower()}"
                    existing_contacts[key] = contact
        except Exception as e:
            print(f"Error loading existing contacts JSON: {str(e)}")
            # If there's an error, start with an empty contacts list
            existing_contacts = {}
    
    # Add new contacts with deduplication
    contacts_added = 0
    for contact in contacts:
        # Skip empty contacts
        if not contact.get('name') or contact.get('name').strip() == "":
            continue
            
        # Create key for deduplication
        key = f"{contact.get('name', '').lower()}_{contact.get('company', '').lower()}"
        
        # Check if this contact already exists
        if key not in existing_contacts:
            # Add source information
            contact["source_frame"] = {
                "frame_number": frame_info.get("frame_number"),
                "timestamp": frame_info.get("timestamp"),
                "filename": frame_info.get("filename"),
                "path": frame_info.get("path")
            }
            
            existing_contacts[key] = contact
            contacts_added += 1
        else:
            # Contact already exists, potentially update with newer information
            # You could add logic here to merge information or add this frame as an additional source
            if "source_frames" not in existing_contacts[key]:
                existing_contacts[key]["source_frames"] = [existing_contacts[key].get("source_frame", {})]
                del existing_contacts[key]["source_frame"]
            
            # Add this frame as an additional source
            existing_contacts[key]["source_frames"].append({
                "frame_number": frame_info.get("frame_number"),
                "timestamp": frame_info.get("timestamp"),
                "filename": frame_info.get("filename"),
                "path": frame_info.get("path")
            })
    
    # Convert dictionary back to list
    contacts_list = list(existing_contacts.values())
    
    # Create the JSON data structure
    json_data = {
        "contacts": contacts_list,
        "last_updated": datetime.now().isoformat()
    }
    
    # Write to JSON file
    with open(contacts_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Contacts saved to {contacts_json_path}. Added {contacts_added} new contacts, total: {len(contacts_list)}")

class KeyFrameGenerator:
    def __init__(self, video_path: str, similarity_threshold: float = 0.8, scale_percent: int = 30):
        """
        Initialize the key frame generator.
        
        Args:
            video_path (str): Path to the input video
            similarity_threshold (float): Threshold for detecting key frames (0.0-1.0)
            scale_percent (int): Percentage to scale the output thumbnails
        """
        self.video_path = video_path
        self.similarity_threshold = similarity_threshold
        self.scale_percent = scale_percent
        
        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate scaled dimensions
        self.scaled_width = int(self.frame_width * self.scale_percent / 100)
        self.scaled_height = int(self.frame_height * self.scale_percent / 100)
        
        print(f"Video loaded: {os.path.basename(video_path)}")
        print(f"Dimensions: {self.frame_width}x{self.frame_height}")
        print(f"Output size: {self.scaled_width}x{self.scaled_height}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
    
    def calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate the difference between two frames.
        
        Args:
            frame1, frame2: Frames to compare
            
        Returns:
            float: Difference score between 0.0 (identical) and 1.0 (completely different)
        """
        # Convert frames to grayscale for faster comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between frames
        diff = cv2.absdiff(gray1, gray2)
        
        # Normalize difference score between 0.0 and 1.0
        return np.sum(diff) / (255.0 * diff.size)
    
    def extract_key_frames(self, output_dir: str, save_json: bool = False) -> List[Tuple[int, float]]:
        """
        Extract key frames from the video based on the similarity threshold.
        
        Args:
            output_dir (str): Directory to save extracted frames
            save_json (bool): Whether to save frame metadata to a JSON file
            
        Returns:
            List[Tuple[int, float]]: List of (frame_number, difference_score) for key frames
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create keyframes subfolder if saving JSON
        keyframes_dir = os.path.join(output_dir, "keyframes")
        if save_json:
            os.makedirs(keyframes_dir, exist_ok=True)
        else:
            keyframes_dir = output_dir
        
        # Initialize variables
        prev_frame = None
        key_frames = []
        frame_count = 0
        start_time = time.time()
        
        # Check for existing JSON data
        json_path = os.path.join(output_dir, "keyframes.json")
        processed_frames = {}
        
        if save_json and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    processed_frames = {item["frame_number"]: item for item in data.get("frames", [])}
                print(f"Loaded {len(processed_frames)} previously processed frames from {json_path}")
            except Exception as e:
                print(f"Error loading existing JSON data: {str(e)}")
        
        print(f"\nExtracting key frames with similarity threshold: {self.similarity_threshold}")
        
        # Process video frames
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Update progress every second or so
            if frame_count % int(self.fps) == 0:
                elapsed_time = time.time() - start_time
                progress = (frame_count / self.total_frames) * 100
                frames_per_second = frame_count / max(elapsed_time, 0.001)
                estimated_time = (self.total_frames - frame_count) / max(frames_per_second, 0.001)
                
                print(f"Progress: {progress:.1f}% ({frame_count}/{self.total_frames} frames)")
                print(f"Processing speed: {frames_per_second:.1f} fps")
                print(f"Estimated time remaining: {estimated_time:.1f} seconds")
            
            # Skip already processed frames if they exist in the JSON
            if frame_count in processed_frames:
                frame_count += 1
                continue
            
            # First frame is always a key frame
            if prev_frame is None:
                prev_frame = frame.copy()
                difference = 1.0  # Max difference for first frame
                key_frames.append((frame_count, difference))
                
                # Save the first frame
                filename = self._save_frame(frame, frame_count, difference, keyframes_dir)
            else:
                # Calculate difference with previous frame
                difference = self.calculate_frame_difference(prev_frame, frame)
                
                # If difference exceeds threshold, this is a key frame
                if difference >= self.similarity_threshold:
                    key_frames.append((frame_count, difference))
                    prev_frame = frame.copy()
                    
                    # Save the key frame
                    filename = self._save_frame(frame, frame_count, difference, keyframes_dir)
                    
                    print(f"Key frame found at {frame_count} with difference: {difference:.4f}")
            
            frame_count += 1
        
        # Release video capture
        self.cap.release()
        
        # If saving to JSON, add the new key frames and write the file
        if save_json:
            # Combine existing data with new key frames
            all_frames = list(processed_frames.values())
            
            # Add new key frames
            for frame_num, diff in key_frames:
                # Skip if already in processed frames
                if frame_num in processed_frames:
                    continue
                    
                timestamp = frame_num / self.fps
                filename = f"frame_{frame_num:06d}_{timestamp:.2f}s_{diff:.4f}.jpg"
                
                frame_data = {
                    "frame_number": frame_num,
                    "timestamp": timestamp,
                    "difference": diff,
                    "filename": filename,
                    "path": os.path.join("keyframes", filename)
                }
                all_frames.append(frame_data)
            
            # Sort by frame number
            all_frames.sort(key=lambda x: x["frame_number"])
            
            # Create the JSON data structure
            json_data = {
                "video_path": self.video_path,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "similarity_threshold": self.similarity_threshold,
                "frames": all_frames
            }
            
            # Write to JSON file
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"Saved metadata for {len(all_frames)} key frames to {json_path}")
        
        print(f"\nExtraction complete! Found {len(key_frames)} new key frames.")
        print(f"Output saved to: {keyframes_dir}")
        
        return key_frames
    
    def _save_frame(self, frame: np.ndarray, frame_number: int, difference: float, output_dir: str) -> str:
        """
        Save a frame as a thumbnail image.
        
        Returns:
            str: The filename of the saved frame
        """
        # Scale the frame to the desired output size
        scaled_frame = cv2.resize(frame, (self.scaled_width, self.scaled_height))
        
        # Format the filename with frame number and difference score
        timestamp = frame_number / self.fps
        filename = f"frame_{frame_number:06d}_{timestamp:.2f}s_{difference:.4f}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Save the frame
        cv2.imwrite(output_path, scaled_frame)
        
        return filename

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract key frames from a video')
    parser.add_argument('--video', '-v', type=str, required=True,
                        help='Path to the input video file')
    parser.add_argument('--output', '-o', type=str, default='keyframes',
                        help='Directory to save extracted frames')
    parser.add_argument('--threshold', '-t', type=float, default=0.8,
                        help='Similarity threshold for detecting key frames (0.0-1.0)')
    parser.add_argument('--scale', '-s', type=int, default=30,
                        help='Percentage to scale the output thumbnails (default: 30%%)')
    parser.add_argument('--analyze', '-a', action='store_true',
                        help='Analyze existing frames without generating new thumbnails')
    
    args = parser.parse_args()
    
    # Create output directory (use video name as subfolder by default)
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = os.path.join(args.output, video_name)
    
    # Check if we're only analyzing existing frames
    if not args.analyze:
        # Initialize and run key frame generator
        generator = KeyFrameGenerator(
            args.video,
            similarity_threshold=args.threshold,
            scale_percent=args.scale
        )
        
        # Extract key frames (always save to JSON)
        key_frames = generator.extract_key_frames(output_dir, save_json=True)
    
    # Analyze key frames if requested
    if args.analyze:
        # Load the JSON data
        json_path = os.path.join(output_dir, "keyframes.json")
        frames_analyzed = 0
        
        if not os.path.exists(json_path):
            print(f"Error: Cannot find JSON file at {json_path}. Run the script without --analyze first.")
            return
        
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            print(f"Found {len(json_data.get('frames', []))} frames in JSON file. Starting analysis...")
            
            # Process each frame that doesn't have analysis yet
            for frame in json_data.get("frames", []):
                # Skip frames that already have analysis
                if "analysis" in frame:
                    continue
                
                # Get the full path to the frame image
                frame_path = os.path.join(output_dir, frame.get("path", ""))
                
                if os.path.exists(frame_path):
                    print(f"Analyzing frame: {frame.get('filename')}")
                    
                    # Extract information from the frame
                    search_query = f"Frame at {frame.get('timestamp', 0):.2f}s"
                    prompt = f"Extract information from video frame at timestamp {frame.get('timestamp', 0):.2f}s"
                    
                    # Call the analysis function with the image path
                    analysis_result = extract_contact_info(search_query, prompt, frame_path)
                    
                    # Add the analysis to the frame data
                    frame["analysis"] = analysis_result
                    frames_analyzed += 1
                    
                    # Save extracted contacts to separate JSON
                    if "contacts" in analysis_result:
                        save_contacts_to_json(output_dir, analysis_result["contacts"], frame)
                else:
                    print(f"Warning: Frame file not found: {frame_path}")
            
            # Save the updated JSON
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"Analysis complete! Analyzed {frames_analyzed} new frames.")
            
        except Exception as e:
            print(f"Error during frame analysis: {str(e)}")

if __name__ == '__main__':
    main() 