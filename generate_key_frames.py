import cv2
import os
import numpy as np
import time
import argparse
from typing import List, Tuple, Dict
from datetime import datetime
import json
import base64
import csv

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

def extract_contact_info(image_path: str) -> Dict:
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
    def __init__(self, video_path: str, center_height_percent: float = 40, overlap_threshold: float = 0.3, debug: bool = True):
        """
        Initialize the key frame generator optimized for scrolling videos.
        
        Args:
            video_path (str): Path to the input video
            center_height_percent (float): Percentage of frame height to use as tracking region (default 40%)
            overlap_threshold (float): How much the tracking region should move before capturing (default 30%)
            debug (bool): Whether to output debug visualization frames
        """
        self.video_path = video_path
        self.center_height_percent = center_height_percent
        self.overlap_threshold = overlap_threshold
        self.debug = debug
        
        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate tracking region dimensions
        self.tracking_height = int(self.frame_height * (center_height_percent / 100))
        self.tracking_y_start = (self.frame_height - self.tracking_height) // 2
        
        print(f"Video loaded: {os.path.basename(video_path)}")
        print(f"Dimensions: {self.frame_width}x{self.frame_height}")
        print(f"Tracking region height: {self.tracking_height}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")

    def _save_frame(self, frame: np.ndarray, frame_number: int, movement: float, output_dir: str, flow: np.ndarray = None) -> str:
        """
        Save a frame as a thumbnail image.
        
        Returns:
            str: The filename of the saved frame
        """
        # Format the filename with frame number and movement amount
        timestamp = frame_number / self.fps
        filename = f"frame_{frame_number:06d}_{timestamp:.2f}s_{movement:.4f}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Save the original frame without any debug visualization
        cv2.imwrite(output_path, frame)
        
        return filename

    def draw_debug_visualization(self, frame: np.ndarray, flow: np.ndarray = None, 
                               movement: float = 0.0, is_scrolling: bool = False,
                               trend_strength: float = 0.0, is_keyframe: bool = False) -> np.ndarray:
        """
        Draw debug visualization showing tracking region and movement.
        
        Args:
            frame: Current video frame
            flow: Optical flow data if available
            movement: Current movement amount
            is_scrolling: Whether consistent scrolling is detected
            trend_strength: Strength of the scrolling trend
            is_keyframe: Whether this frame was selected as a keyframe
            
        Returns:
            np.ndarray: Frame with debug visualization
        """
        debug_frame = frame.copy()
        
        # If this is a keyframe, draw a thick red border around the entire frame
        if is_keyframe:
            border_thickness = 10
            h, w = debug_frame.shape[:2]
            cv2.rectangle(
                debug_frame,
                (0, 0),
                (w-1, h-1),
                (0, 0, 255),  # Red
                border_thickness
            )
        
        # Draw tracking region boundaries
        color = (0, 255, 0) if is_scrolling else (0, 165, 255)  # Green if scrolling, orange if not
        cv2.rectangle(
            debug_frame,
            (0, self.tracking_y_start),
            (self.frame_width, self.tracking_y_start + self.tracking_height),
            color,
            2
        )
        
        # If we have flow data, visualize it
        if flow is not None:
            # Draw flow arrows every N pixels
            step = 32
            max_flow = 0.0
            significant_movements = []  # Store positions of significant movements
            
            # First pass to calculate statistics
            for y_offset in range(0, self.tracking_height, step):
                for x in range(0, self.frame_width, step):
                    flow_y = y_offset
                    frame_y = self.tracking_y_start + y_offset
                    
                    if flow_y < flow.shape[0] and x < flow.shape[1]:
                        fx, fy = flow[flow_y, x]
                        max_flow = max(max_flow, abs(fy))
                        if abs(fy) > 0.5:  # Significant movement threshold
                            significant_movements.append((x, frame_y, fx, fy))
            
            # Draw significant movements with large, obvious arrows
            for x, frame_y, fx, fy in significant_movements:
                # Draw a bright red arrow with thick line
                cv2.arrowedLine(
                    debug_frame,
                    (x, frame_y),
                    (int(x + fx), int(frame_y + fy)),
                    (0, 0, 255),  # Red
                    3,  # Thicker line
                    tipLength=0.5
                )
                
                # Draw large circle at the base
                cv2.circle(
                    debug_frame,
                    (x, frame_y),
                    6,  # Larger radius
                    (0, 0, 255),  # Red
                    -1  # Filled
                )
                
                # Draw movement magnitude
                cv2.putText(
                    debug_frame,
                    f"{abs(fy):.2f}",
                    (x + 5, frame_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
            
            # Add flow statistics with highlight for trigger condition
            stats_color = (0, 0, 255) if len(significant_movements) > 10 else (255, 255, 255)
            cv2.putText(
                debug_frame,
                f"Significant Movements: {len(significant_movements)} points",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                stats_color,
                2
            )
            
            # Add max flow with highlight
            max_flow_color = (0, 0, 255) if max_flow > 0.5 else (255, 255, 255)
            cv2.putText(
                debug_frame,
                f"Max Flow: {max_flow:.4f}",
                (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                max_flow_color,
                2
            )
        
        # Add movement text with highlight for trigger condition
        movement_color = (0, 0, 255) if movement >= self.overlap_threshold else (255, 255, 255)
        cv2.putText(
            debug_frame,
            f"Accumulated Movement: {movement:.4f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            movement_color,
            2
        )
        
        # Add tracking region size text
        cv2.putText(
            debug_frame,
            f"Track Region: {self.tracking_height}px",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        
        # Add scrolling trend info
        status = "SCROLLING" if is_scrolling else "STATIC"
        cv2.putText(
            debug_frame,
            f"Status: {status} ({trend_strength:.4f})",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        
        # If this is a keyframe, add text indicating why it was selected
        if is_keyframe:
            cv2.putText(
                debug_frame,
                "KEYFRAME",
                (debug_frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        return debug_frame

    def get_tracking_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the center tracking region from a frame.
        
        Args:
            frame: Full video frame
            
        Returns:
            np.ndarray: Center region used for tracking
        """
        return frame[self.tracking_y_start:self.tracking_y_start + self.tracking_height, :]
    
    def calculate_region_movement(self, region1: np.ndarray, region2: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate how much the tracking region has moved between frames.
        
        Args:
            region1, region2: Tracking regions to compare
            
        Returns:
            Tuple[float, np.ndarray]: Movement amount (0-1) and movement direction vector
        """
        # Convert regions to grayscale
        gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate average vertical movement
        avg_movement = np.mean(flow[:, :, 1])  # Use y-component of flow
        
        # Normalize movement relative to region height
        movement_ratio = abs(avg_movement) / self.tracking_height
        
        return movement_ratio, flow

    def detect_scrolling_trend(self, movements: List[float], window_size: int = 5) -> Tuple[bool, float]:
        """
        Use two-pointer algorithm to detect consistent scrolling trend.
        
        Args:
            movements: List of recent movement values
            window_size: Size of the sliding window for trend detection
            
        Returns:
            Tuple[bool, float]: Whether consistent scrolling is detected and the trend strength
        """
        if len(movements) < window_size:
            return False, 0.0
            
        # Use two pointers to analyze movement pattern
        slow = len(movements) - window_size
        fast = len(movements) - 1
        
        # Calculate trend metrics
        total_movement = 0.0
        direction_changes = 0
        prev_direction = None
        max_movement = 0.0  # Track maximum movement in window
        
        while slow < fast:
            # Calculate movement direction between adjacent points
            movement_diff = movements[slow + 1] - movements[slow]
            current_direction = 1 if movement_diff > 0 else -1 if movement_diff < 0 else 0
            
            # Track direction changes
            if prev_direction is not None and current_direction != prev_direction and current_direction != 0:
                direction_changes += 1
            
            prev_direction = current_direction
            total_movement += abs(movement_diff)
            max_movement = max(max_movement, abs(movements[slow + 1]))
            slow += 1
        
        # Calculate trend strength metrics
        avg_movement = total_movement / (window_size - 1)
        consistency = 1.0 - (direction_changes / (window_size - 1))
        
        # Detect if we have consistent scrolling
        # Lower consistency requirement if we detect a very strong movement
        consistency_threshold = 0.4 if max_movement > 0.1 else 0.6
        movement_threshold = 0.005  # More sensitive to small movements
        
        is_scrolling = (consistency > consistency_threshold and avg_movement > movement_threshold) or max_movement > 0.1
        trend_strength = max(consistency * avg_movement, max_movement)
        
        return is_scrolling, trend_strength

    def calculate_frame_overlap(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate the overlap ratio between two frames using feature matching.
        
        Args:
            frame1, frame2: Frames to compare
            
        Returns:
            float: Overlap ratio between 0.0 (no overlap) and 1.0 (identical)
        """
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # If no features found, return 0 overlap
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0
        
        # Create FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Find matches
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
        
        # Calculate overlap ratio based on number of good matches
        overlap_ratio = len(good_matches) / max(len(kp1), len(kp2))
        
        return min(1.0, overlap_ratio)

    def filter_redundant_keyframes(self, keyframes_dir: str, max_overlap: float = 0.6) -> List[str]:
        """
        Filter out redundant keyframes that have too much overlap with their neighbors.
        
        Args:
            keyframes_dir: Directory containing keyframes
            max_overlap: Maximum allowed overlap ratio between consecutive frames
            
        Returns:
            List[str]: List of files to keep
        """
        # Get list of keyframe files sorted by frame number
        keyframe_files = sorted([f for f in os.listdir(keyframes_dir) if f.startswith('frame_')])
        
        if not keyframe_files:
            return []
        
        # Initialize list of frames to keep
        frames_to_keep = [keyframe_files[0]]  # Always keep first frame
        prev_frame = cv2.imread(os.path.join(keyframes_dir, keyframe_files[0]))
        
        print(f"\nFiltering redundant keyframes...")
        print(f"Starting with {len(keyframe_files)} keyframes")
        
        # Compare consecutive frames
        for i in range(1, len(keyframe_files)):
            current_file = keyframe_files[i]
            current_frame = cv2.imread(os.path.join(keyframes_dir, current_file))
            
            # Calculate overlap with previous kept frame
            overlap = self.calculate_frame_overlap(prev_frame, current_frame)
            
            # If overlap is below threshold, keep this frame
            if overlap < max_overlap:
                frames_to_keep.append(current_file)
                prev_frame = current_frame
                print(f"Keeping frame {current_file} (overlap: {overlap:.3f})")
            else:
                print(f"Removing frame {current_file} (overlap: {overlap:.3f})")
        
        print(f"Filtered to {len(frames_to_keep)} keyframes")
        return frames_to_keep

    def extract_key_frames(self, output_dir: str, save_json: bool = False, filter_redundant: bool = True) -> List[Tuple[int, float]]:
        """
        Extract key frames from the video based on tracking region movement.
        
        Args:
            output_dir: Directory to save extracted frames
            save_json: Whether to save frame metadata to a JSON file
            filter_redundant: Whether to perform redundant frame filtering
            
        Returns:
            List[Tuple[int, float]]: List of (frame_number, movement_amount) for key frames
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        if self.debug:
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
        
        keyframes_dir = os.path.join(output_dir, "keyframes")
        if save_json:
            os.makedirs(keyframes_dir, exist_ok=True)
        else:
            keyframes_dir = output_dir
        
        # Initialize variables
        prev_region = None
        key_frames = []
        frame_count = 0
        accumulated_movement = 0.0
        start_time = time.time()
        last_flow = None
        recent_movements = []  # Track recent movements for trend detection
        
        # Process video frames
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Get current tracking region
            current_region = self.get_tracking_region(frame)
            
            # First frame is always a key frame
            if prev_region is None:
                prev_region = current_region.copy()
                movement = 1.0  # Max movement for first frame
                key_frames.append((frame_count, movement))
                recent_movements.append(movement)
                
                # Save the first frame without debug info
                self._save_frame(frame, frame_count, movement, keyframes_dir)
                
                if self.debug:
                    debug_frame = self.draw_debug_visualization(
                        frame, None, movement, 
                        is_keyframe=True
                    )
                    cv2.imwrite(
                        os.path.join(debug_dir, f"debug_{frame_count:06d}.jpg"),
                        debug_frame
                    )
            else:
                # Calculate movement between regions
                movement, flow = self.calculate_region_movement(prev_region, current_region)
                recent_movements.append(movement)
                last_flow = flow
                
                # Calculate flow statistics
                max_flow = 0.0
                significant_movements = 0
                step = 32
                
                for y_offset in range(0, self.tracking_height, step):
                    for x in range(0, self.frame_width, step):
                        flow_y = y_offset
                        if flow_y < flow.shape[0] and x < flow.shape[1]:
                            fx, fy = flow[flow_y, x]
                            max_flow = max(max_flow, abs(fy))
                            if abs(fy) > 0.8:  # Increased threshold for significant movements
                                significant_movements += 1
                
                # Detect scrolling trend
                is_scrolling, trend_strength = self.detect_scrolling_trend(recent_movements)
                
                # Update accumulated movement based on trend
                if is_scrolling:
                    accumulated_movement += movement * trend_strength
                else:
                    # Decay accumulated movement when no clear trend
                    accumulated_movement *= 0.8
                
                # Determine if this should be a key frame
                should_capture = False
                capture_reason = ""
                
                # Case 1: Strong individual movement detected
                if max_flow > 0.8:  # Increased threshold for strong movement
                    should_capture = True
                    capture_reason = f"Strong movement detected: {max_flow:.4f}"
                
                # Case 2: Many significant movement points
                elif significant_movements > 8:  # Slightly reduced threshold since movements are more significant
                    should_capture = True
                    capture_reason = f"Multiple movement points: {significant_movements}"
                
                # Case 3: Accumulated movement threshold reached
                elif accumulated_movement >= self.overlap_threshold:
                    should_capture = True
                    capture_reason = f"Accumulated movement: {accumulated_movement:.4f}"
                
                if should_capture:
                    key_frames.append((frame_count, max(max_flow, accumulated_movement)))
                    prev_region = current_region.copy()
                    
                    # Save the key frame without debug info
                    self._save_frame(frame, frame_count, accumulated_movement, keyframes_dir)
                    
                    print(f"Key frame found at {frame_count} - {capture_reason}")
                
                if self.debug:
                    debug_frame = self.draw_debug_visualization(
                        frame, flow, accumulated_movement, 
                        is_scrolling=is_scrolling,
                        trend_strength=trend_strength,
                        is_keyframe=should_capture
                    )
                    cv2.imwrite(
                        os.path.join(debug_dir, f"debug_{frame_count:06d}.jpg"),
                        debug_frame
                    )
                
                # Reset accumulated movement if we captured a frame
                if should_capture:
                    accumulated_movement = 0.0
                
                # Keep recent movements list at window size
                if len(recent_movements) > 10:
                    recent_movements.pop(0)
            
            frame_count += 1
            
            # Update progress
            if frame_count % int(self.fps) == 0:
                elapsed_time = time.time() - start_time
                progress = (frame_count / self.total_frames) * 100
                frames_per_second = frame_count / max(elapsed_time, 0.001)
                estimated_time = (self.total_frames - frame_count) / max(frames_per_second, 0.001)
                
                print(f"Progress: {progress:.1f}% ({frame_count}/{self.total_frames} frames)")
                print(f"Processing speed: {frames_per_second:.1f} fps")
                print(f"Estimated time remaining: {estimated_time:.1f} seconds")
                print(f"Current accumulated movement: {accumulated_movement:.4f}")
        
        # Release video capture
        self.cap.release()
        
        print(f"\nInitial extraction complete! Found {len(key_frames)} key frames.")
        print(f"Output saved to: {keyframes_dir}")
        
        # Phase 2: Filter redundant frames if requested
        if filter_redundant:
            frames_to_keep = self.filter_redundant_keyframes(keyframes_dir)
            
            # Remove files that didn't make the cut
            for filename in os.listdir(keyframes_dir):
                if filename.startswith('frame_') and filename not in frames_to_keep:
                    os.remove(os.path.join(keyframes_dir, filename))
            
            # Update key_frames list to match kept frames
            kept_frame_numbers = [int(f.split('_')[1]) for f in frames_to_keep]
            key_frames = [(fn, mv) for fn, mv in key_frames if fn in kept_frame_numbers]
            
            print(f"After filtering: {len(key_frames)} unique key frames")
        
        if self.debug:
            print(f"Debug frames saved to: {debug_dir}")
        
        return key_frames

def get_processed_frames(csv_path: str) -> set:
    """
    Get the set of already processed frames from the contacts CSV file.
    
    Args:
        csv_path: Path to the contacts CSV file
        
    Returns:
        set: Set of processed frame filenames
    """
    processed_frames = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('source_frame'):
                        processed_frames.add(row['source_frame'])
                        print(f"Found already processed frame in contacts.csv: {row['source_frame']}")
        except Exception as e:
            print(f"Error reading contacts CSV: {str(e)}")
    return processed_frames

def save_contacts_to_csv(contacts: List[Dict], output_path: str, mode: str = 'w'):
    """
    Save extracted contact information to a CSV file.
    
    Args:
        contacts: List of contact dictionaries
        output_path: Path to save the CSV file
        mode: File open mode ('w' for write, 'a' for append)
    """
    if not contacts:
        print("No contacts to save")
        return
        
    # Define CSV fields
    fields = ['name', 'title', 'company', 'email', 'phone', 'website', 'source_frame']
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(output_path) and mode == 'a'
    
    with open(output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerows(contacts)
    
    print(f"\nSaved {len(contacts)} contacts to {output_path}")

def analyze_keyframes(keyframes_dir: str, scale_percent: int = 30, csv_path: str = None) -> List[Dict]:
    """
    Analyze all keyframe images in the directory to extract contact information.
    Tracks progress in the contacts CSV file and saves results incrementally.
    
    Args:
        keyframes_dir: Directory containing keyframe images
        scale_percent: Percentage to scale images before analysis
        csv_path: Path to the contacts CSV file
        
    Returns:
        List[Dict]: List of extracted contact information
    """
    # Get list of keyframe files
    keyframe_files = sorted([f for f in os.listdir(keyframes_dir) if f.startswith('frame_')])
    all_contacts = []
    
    # Get already processed frames from CSV
    processed_frames = get_processed_frames(csv_path) if csv_path else set()
    
    print(f"\nAnalyzing {len(keyframe_files)} keyframes...")
    print(f"Found {len(processed_frames)} already processed frames")
    
    for i, filename in enumerate(keyframe_files, 1):
        # Skip if already processed
        if filename in processed_frames:
            print(f"Skipping already processed frame {i}/{len(keyframe_files)}: {filename}")
            continue
            
        filepath = os.path.join(keyframes_dir, filename)
        
        # Read and scale the image
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error reading image: {filepath}")
            continue
            
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        scaled_img = cv2.resize(img, (width, height))
        
        # Save scaled image temporarily
        temp_path = os.path.join(keyframes_dir, f"temp_{filename}")
        cv2.imwrite(temp_path, scaled_img)
        
        print(f"Processing frame {i}/{len(keyframe_files)}: {filename}")
        
        try:
            # Extract contact information
            result = extract_contact_info(temp_path)
            
            # Add source information to each contact
            contacts = []
            for contact in result.get('contacts', []):
                contact['source_frame'] = filename
                contacts.append(contact)
            
            # Save contacts immediately after processing
            if contacts and csv_path:
                save_contacts_to_csv(contacts, csv_path, mode='a')
            
            all_contacts.extend(contacts)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return all_contacts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract key frames from a video')
    parser.add_argument('--video', '-v', type=str,
                        help='Path to the input video file')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='Overlap threshold for detecting key frames (0.0-1.0)')
    parser.add_argument('--max-overlap', '-m', type=float, default=0.6,
                        help='Maximum allowed overlap between consecutive keyframes (0.0-1.0)')
    parser.add_argument('--scale', '-s', type=int, default=30,
                        help='Percentage to scale the output thumbnails (default: 30%%)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable redundant frame filtering')
    parser.add_argument('--analyze', '-a', action='store_true',
                        help='Analyze existing keyframes to extract contact information')
    
    args = parser.parse_args()
    
    if not args.video and not args.analyze:
        parser.error("Either --video or --analyze must be specified")
    
    # Create output directory based on video name
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = os.path.join(os.path.dirname(args.video), video_name)
    
    # Process video and extract keyframes if not in analyze-only mode
    if not args.analyze:
        # Initialize and run key frame generator
        generator = KeyFrameGenerator(
            args.video,
            overlap_threshold=args.threshold
        )
        
        # Extract key frames with filtering
        key_frames = generator.extract_key_frames(
            output_dir, 
            save_json=True,
            filter_redundant=not args.no_filter
        )
    
    # Analyze keyframes if requested
    if args.analyze:
        keyframes_dir = os.path.join(output_dir, "keyframes")
        if not os.path.exists(keyframes_dir):
            keyframes_dir = output_dir
            
        if not os.path.exists(keyframes_dir):
            print(f"Error: Keyframes directory not found: {keyframes_dir}")
            return
            
        # Set up CSV path
        csv_path = os.path.join(output_dir, "contacts.csv")
            
        # Extract contact information from keyframes
        contacts = analyze_keyframes(keyframes_dir, args.scale, csv_path)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "contacts.csv")
        save_contacts_to_csv(contacts, csv_path)

if __name__ == '__main__':
    main() 