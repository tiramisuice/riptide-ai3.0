import os
import cv2
import asyncio
import base64
from openai import AsyncOpenAI  # <-- switched to asyncopenai
from dotenv import load_dotenv

# ─── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
VIDEO_PATH = "girl_drown.mp4"  # your local video file
FRAME_DELAY = 0.5  # seconds between frames
API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=API_KEY)

# ─── Helpers ────────────────────────────────────────────────────────────────────


def frame_to_data_url(frame):
    """
    Encode an OpenCV BGR frame (numpy array) into a JPEG Data URL.
    """
    print("[debug] Encoding frame to JPEG data URL...")
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
    """
    Send the image (as a Data URL) to OpenAI's Responses API (gpt-4o)
    and return a summary of what's in the photo.
    """
    print(f"[debug] [{index}] Sending image to GPT-4o for analysis...")
    prompt = (
        "You are a lifeguard tasked with identifying potentially drowning people in a pool.\n"
        "Some rules of thumb: False positives are infinitely better than false negatives.\n"
        "Drowning people often go unnoticed because they look like regular people in a pool.\n"
        "Most often they don't thrash around, and silently sink in the pool.\n"
        "Therefore you must be very careful in your analysis, as lives are at stake.\n"
        "Right now you must carefully take notes on photos that were taken every {FRAME_DELAY} seconds.\n"
        "In super concise form, record coordinates of each person in the photo\n"
        "and their arm pose, how submerged they are in the water (including whether head is submerged)\n"
        "and their position relative to the water (above, below, or in the water)\n"
    )

    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": data_url, "detail": "high"},
    ]
    response = await client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
    )
    summary = response.output_text
    print(f"[debug] [{index}] Received summary (length {len(summary)}).")
    return summary


# ─── Main ───────────────────────────────────────────────────────────────────────


async def main():
    # 1. Extract frames in memory & convert to Data URLs
    print("[debug] Starting frame extraction...")
    data_urls = []
    for frame in extract_frames_in_memory(VIDEO_PATH, FRAME_DELAY):
        url = frame_to_data_url(frame)
        data_urls.append(url)
    total = len(data_urls)
    print(f"[debug] Prepared {total} frame Data URLs in memory.")

    # 2. Analyze all frames in parallel
    print("[debug] Launching analysis tasks...")
    tasks = [analyze_photo_dataurl(url, i) for i, url in enumerate(data_urls)]
    summaries = await asyncio.gather(*tasks)

    # 3. Print each frame's summary
    print("[debug] All analyses complete. Printing summaries...\n")
    for i, text in enumerate(summaries):
        print(f"[Frame {i:03d}]: {text}\n")


if __name__ == "__main__":
    print("[debug] Running main()")
    asyncio.run(main())
    print("[debug] Done.")
