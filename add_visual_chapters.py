def create_output_video(input_video_path, output_video_path, chapters):
    """
    Create a new video with visual chapter markers.
    """
    # Open the input video
    video = VideoFileClip(input_video_path)
    
    # Get video dimensions
    width, height = video.size
    
    # Create a list to store all clips
    clips = []
    
    # Process each chapter
    for i, chapter in enumerate(chapters):
        start_time = chapter['start_time']
        end_time = chapter['end_time'] if i < len(chapters) - 1 else video.duration
        title = chapter['title']
        
        # Extract the clip for this chapter
        clip = video.subclip(start_time, end_time)
        
        # Create a text clip for the chapter title
        txt_clip = TextClip(title, fontsize=24, color='white', bg_color='rgba(0,0,0,0.5)', 
                           size=(width, None), method='caption', align='center')
        txt_clip = txt_clip.set_position(('center', 50)).set_duration(5)  # Show for 5 seconds
        
        # Composite the text over the video for the first 5 seconds of the chapter
        # Explicitly preserve the audio from the original clip
        chapter_clip = CompositeVideoClip([clip, txt_clip])
        chapter_clip = chapter_clip.set_duration(clip.duration)
        
        # Ensure audio is preserved
        chapter_clip = chapter_clip.set_audio(clip.audio)
        
        # Add to the list of clips
        clips.append(chapter_clip)
    
    # Concatenate all clips
    final_clip = concatenate_videoclips(clips)
    
    # Write the result to a file with the original audio
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac') 