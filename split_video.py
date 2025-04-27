import json
import os
import cv2
import asyncio
import base64
import re
import numpy as np
from openai import AsyncOpenAI     # <-- switched to asyncopenai
from dotenv import load_dotenv
from typing import List, Union

# ─── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
VIDEO_PATH    = "girl_drown.mp4"                # your local video file
FRAME_DELAY   = 0.5                             # seconds between frames
API_KEY       = os.getenv("OPENAI_API_KEY")
client        = AsyncOpenAI(api_key=API_KEY)

# ─── Helpers ────────────────────────────────────────────────────────────────────

def downscale_image_for_openai(frame, max_width=640, max_height=480):
    """
    Downscale an image to a reasonable size for OpenAI API calls to reduce costs.
    """
    height, width = frame.shape[:2]
    
    # If image is already smaller than max dimensions, return as is
    if width <= max_width and height <= max_height:
        return frame
    
    # Calculate the scaling factor
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    print(f"[debug] Resized image from {width}x{height} to {new_width}x{new_height}")
    return resized_image

def downscale_data_url(data_url, max_width=640, max_height=480):
    """
    Downscale an image in data URL format for OpenAI API calls to reduce costs.
    """
    # Check if data URL needs processing
    pattern = r'data:image\/([a-zA-Z]+);base64,(.+)'
    match = re.match(pattern, data_url)
    
    if not match:
        print("[debug] Not a valid data URL format, returning as is")
        return data_url
        
    img_format, base64_data = match.groups()
    
    # Decode base64 to image
    img_data = base64.b64decode(base64_data)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        print("[debug] Failed to decode image from data URL, returning as is")
        return data_url
    
    # Get current dimensions
    height, width = img.shape[:2]
    
    # Check if downscaling is needed
    if width <= max_width and height <= max_height:
        print("[debug] Image already within size limits, no resizing needed")
        return data_url
    
    # Downscale the image
    resized_img = downscale_image_for_openai(img, max_width, max_height)
    
    # Convert back to data URL
    success, buffer = cv2.imencode(f'.{img_format.lower()}', resized_img)
    if not success:
        print("[debug] Failed to encode resized image, returning original data URL")
        return data_url
    
    b64_resized = base64.b64encode(buffer).decode('ascii')
    resized_data_url = f'data:image/{img_format};base64,{b64_resized}'
    
    print(f"[debug] Resized data URL from approx. {len(data_url)} to {len(resized_data_url)} chars")
    return resized_data_url

def frame_to_data_url(frame, for_openai=True):
    """
    Encode an OpenCV BGR frame (numpy array) into a JPEG Data URL.
    """
    print("[debug] Encoding frame to JPEG data URL...")
    
    # Downscale if needed for OpenAI
    if for_openai:
        frame = downscale_image_for_openai(frame)
    
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(buffer).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    print("[debug] Frame encoded, length:", len(data_url))
    return data_url

def extract_frames_in_memory(video_path, frame_delay):
    """
    Yield frames (as numpy arrays) from `video_path` every `frame_delay` seconds.
    """
    print(f"[debug] Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError("Could not read FPS from video")
    print(f"[debug] Video FPS: {fps}")
    
    interval = max(int(round(fps * frame_delay)), 1)
    print(f"[debug] Frame interval (in frames): {interval}")

    idx = 0
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[debug] End of video stream reached.")
            break
        if idx % interval == 0:
            print(f"[debug] Yielding frame #{count} (video frame idx: {idx})")
            yield frame
            count += 1
        idx += 1

    cap.release()
    print(f"[debug] Released video capture, total frames yielded: {count}")

# ─── Async Photo Analyzer ───────────────────────────────────────────────────────

async def analyze_photo_dataurl(data_url: str, index: int | None = None) -> str:
    # await asyncio.sleep(5)
    # return '{"test": "Temporarily disabled"}'
    """
    Send the image (as a Data URL) to OpenAI's Responses API (gpt-4o)
    and return a summary of what's in the photo.
    """
    print(f"[debug] [{index}] Sending image to GPT-4o for analysis...")
    
    # Downscale the image in the data URL if needed
    data_url = downscale_data_url(data_url)
    
    prompt = """
    You are an AI model for training purposes (not a real lifeguard).
    Task: Given a video clip of someone in water, classify their behavior as Swimming or Drowning using the criteria below. For each person you label, list the specific cues you observed.

    Identification Criteria
    1. Body Orientation & Motion

    Swimming: Torso stays horizontal and moves smoothly through the water.

    Drowning: Body becomes nearly vertical, hips stay low, and may bob up and down without forward tilt.

    2. Arm Movement

    Swimming: Arms execute coordinated full strokes (e.g. freestyle, breaststroke), pushing water backward.

    Drowning: Arms extend out to the sides and press down on the surface—like climbing an invisible ladder—instead of stroking above the head.

    3. Leg Action & Forward Progress

    Swimming: Legs kick steadily (flutter or whip kick) in sync with the arms, producing clear forward movement.

    Drowning: Legs kick weakly or hang limply; despite frantic arm presses, there is little to no forward progress.

    4. Head Position & Breathing

    Swimming: Head rotates smoothly to breathe, lifting above the surface in a regular rhythm; eyes are open and focused.

    Drowning: Head tilts back (chin up) so the mouth bobs at or just below the waterline, gasping rapidly—too preoccupied with breathing to call for help.

    5. Facial & Hair Cues

    Victims may have hair plastered over the face and cannot brush it away.

    Eyes may appear glassy, closed, or unfocused.

    6. Behavioral Signs & Sound

    Swimming: You'll hear audible splashes, exhalations, talking, or purposeful waving.

    Drowning: The struggle is almost always silent—there's no calling out or waving because the body's reflex to breathe dominates.

    7. Duration

    Swimming: Patterns of stroke and breathing repeat reliably and can be sustained indefinitely.

    Drowning: Panic response is brief (about 20–60 seconds) before the person submerges.

    Step-by-Step Identification
    Assess Body Line:

    If the torso is horizontal and gliding → Swimming

    If the body is vertical and bobbing → Drowning

    Watch the Arms:

    Full, rhythmic strokes pulling water backward → Swimming

    Lateral pressing motions at the water's surface → Drowning

    Check the Legs:

    Strong, timed kicks driving forward movement → Swimming

    Weak or absent kicks with no net motion → Drowning

    Observe Breathing:

    Lift, inhale, submerge in a steady cycle → Swimming

    Rapid, panicked gasps at the waterline → Drowning

    Listen and Look for Sound:

    Splashing, talking, or shouting → Swimming

    Silence or only faint splashes → Drowning

    Measure Progress:

    Moves toward safety or across a lap → Swimming

    Drifts in place or stalls despite effort → Drowning

    Note Timing:

    Repeating stroke patterns → Swimming

    Brief, frantic struggle then sinking → Drowning

    Quick "Red Flags" to Trigger a Drowning Alert
    Nearly vertical posture, body bobbing

    Mouth at waterline, rapid gasping

    Arms pressing down rather than stroking

    Legs inactive or kicking weakly

    Silent struggle with minimal splash

    Hair covering the face; glassy or closed eyes

    Use these cues to label each swimmer in the clip.
    """

    content = [
        {"type": "input_text",  "text": prompt},
        {"type": "input_image", "image_url": data_url, "detail": "high"},
    ]
    response = await client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    summary = response.output_text
    print(f"[debug] [{index}] Received summary (length {len(summary)}).")
    return summary

async def summarize_drowning_likelihood(history: str) -> str:
    """
    Accept past ~40 snapshot summary information as string
    to determine likelihood that anybody could be drowning.
    You are not a lifeguard, but rather pretend you're an additional safety
    mechanism that can look at history and help determine if there's
    a chance of somebody drowning.

    Output types:
    Not detected
    Potential drowning
    High-risk drowning

    Some rules of thumb: False positives are infinitely better than false negatives.
    Drowning people often go unnoticed because they look like regular people in a pool.
    Usually they don't thrash around, and silently sink in the pool.

    Input a string combination of JSON objects with the following format
    containing snapshot summary data of multiple people. The objects don't need
    to be in exact same structure.
    """

    # await asyncio.sleep(8)
    # return "Potential drowning"

    prompt = f"""

    Summarize super concisely from the past 40 snapshots (over ~20 seconds)
    If a non-negligible amount of the snapshots say drowning, then say "ALERT DROWNING RISK"
    or something similar IN ALL CAPS FOR EMPHASIS, with "ALERT" is must include substring.
    Otherwise, say "no drowning detected".

    {history}
    """

    content = [
        {"type": "input_text",  "text": prompt},
    ]
    response = await client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    likelihood = response.output_text
    print(f"[debug] Summarize Drowning Likelihood (length {len(likelihood)}).")
    return likelihood


async def get_rectangles(data_url: str) -> List[Union[tuple[int, int, int, int], list[int, int, int, int]]]:
    # Downscale the image in the data URL if needed
    data_url = downscale_data_url(data_url)
    
    prompt = """You are an AI model for training purposes (not a real lifeguard).
    Task: Given a video clip of someone in water, classify their behavior as Swimming or Drowning using the criteria below. For each person you label, list the specific cues you observed, and determine whether they look drowning or just swimming, based on previous observation too

    Identification Criteria
    1. Body Orientation & Motion

    Swimming: Torso stays horizontal and moves smoothly through the water.

    Drowning: Body becomes nearly vertical, hips stay low, and may bob up and down without forward tilt.

    2. Arm Movement

    Swimming: Arms execute coordinated full strokes (e.g. freestyle, breaststroke), pushing water backward.

    Drowning: Arms extend out to the sides and press down on the surface—like climbing an invisible ladder—instead of stroking above the head.

    3. Leg Action & Forward Progress

    Swimming: Legs kick steadily (flutter or whip kick) in sync with the arms, producing clear forward movement.

    Drowning: Legs kick weakly or hang limply; despite frantic arm presses, there is little to no forward progress.

    4. Head Position & Breathing

    Swimming: Head rotates smoothly to breathe, lifting above the surface in a regular rhythm; eyes are open and focused.

    Drowning: Head tilts back (chin up) so the mouth bobs at or just below the waterline, gasping rapidly—too preoccupied with breathing to call for help.

    5. Facial & Hair Cues

    Victims may have hair plastered over the face and cannot brush it away.

    Eyes may appear glassy, closed, or unfocused.

    7. Duration

    Swimming: Patterns of stroke and breathing repeat reliably and can be sustained indefinitely.

    Drowning: Panic response is brief (about 20–60 seconds) before the person submerges.

    Step-by-Step Identification
    Assess Body Line:

    If the torso is horizontal and gliding → Swimming

    If the body is vertical and bobbing → Drowning

    Watch the Arms:

    Full, rhythmic strokes pulling water backward → Swimming

    Lateral pressing motions at the water’s surface → Drowning

    Check the Legs:

    Strong, timed kicks driving forward movement → Swimming

    Weak or absent kicks with no net motion → Drowning

    Observe Breathing:

    Lift, inhale, submerge in a steady cycle → Swimming

    Rapid, panicked gasps at the waterline → Drowning

    Listen and Look for Sound:

    Splashing, talking, or shouting → Swimming

    Silence or only faint splashes → Drowning

    Measure Progress:

    Moves toward safety or across a lap → Swimming

    Drifts in place or stalls despite effort → Drowning

    Note Timing:

    Repeating stroke patterns → Swimming

    Brief, frantic struggle then sinking → Drowning

    Quick “Red Flags” to Trigger a Drowning Alert
    Nearly vertical posture, body bobbing

    Mouth at waterline, rapid gasping

    Arms pressing down rather than stroking

    Legs inactive or kicking weakly

    Silent struggle with minimal splash

    Hair covering the face; glassy or closed eyes

    Use these cues to label each swimmer in the clip.

    """
    content = [
        {"type": "input_text",  "text": prompt},
        {"type": "input_image", "image_url": data_url, "detail": "high"},
    ]
    response = await client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    print(response.output_text)
    rects = json.loads(response.output_text)
    print(f"[debug] get_rectangles {rects}")
    return rects

def draw_rectangles(
    img_path: str,
    rects: List[Union[tuple[int, int, int, int], list[int, int, int, int]]]
) -> None:
    """
    Load an image, draw green rectangles of thickness 2, and overwrite the original file.

    Args:
        img_path: Path to the image file.
        rects: List of 4-element tuples or lists, each as (x1, y1, x2, y2).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {img_path!r}")

    for rect in rects:
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(img_path, img)



# ─── Main ───────────────────────────────────────────────────────────────────────

async def main():
    # 1. Extract frames in memory & convert to Data URLs
    print("[debug] Starting frame extraction...")
    data_urls = []
    for frame in extract_frames_in_memory(VIDEO_PATH, FRAME_DELAY):
        # Use the updated frame_to_data_url with for_openai=True for OpenAI calls
        url = frame_to_data_url(frame, for_openai=True)
        data_urls.append(url)
    total = len(data_urls)
    print(f"[debug] Prepared {total} frame Data URLs in memory.")

    # 2. Analyze all frames in parallel
    print("[debug] Launching analysis tasks...")
    tasks = [
        analyze_photo_dataurl(url, i)
        for i, url in enumerate(data_urls)
    ]
    summaries = await asyncio.gather(*tasks)

    # 3. Print each frame's summary
    print("[debug] All analyses complete. Printing summaries...\n")
    for i, text in enumerate(summaries):
        print(f"[Frame {i:03d}]: {text}\n")

async def main_rectangles():
    image_path = 'output_photos/frame_000020.jpg'
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    
    # Downscale for OpenAI
    img = downscale_image_for_openai(img)
    
    # Convert to data URL
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(buffer).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"
    print(f"[debug] Data URL: {data_url[:100]}... (length: {len(data_url)})")

    # get rectangles
    rects = await get_rectangles(data_url)
    print(f"[debug] Rectangles: {rects}")

    # draw rectangles
    draw_rectangles(image_path.removesuffix('.jpg')+'_rects.jpg', rects)

if __name__ == "__main__":
    # print("[debug] Running main()")
    # asyncio.run(main())
    # print("[debug] Done.")

    # rectangles = [(155, 197, 576, 490)]
    # draw_rectangles("output_photos/20250426-225424_a5ec95d7.jpg", rectangles)

    asyncio.run(main_rectangles())