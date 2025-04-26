# --- Standard Library Imports ---
import logging
import os
import json
import re
import time
import traceback
import uuid
from typing import Any, Dict, AsyncGenerator, List, Optional, Tuple, Union
import asyncio
import tempfile
import warnings
import subprocess
import wave
import site
import sys
import shutil

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()

# --- Simli Import ---
try:
    from simli import SimliClient, SimliConfig
    logger.info("Simli SDK imported.")
    SIMLI_SDK_AVAILABLE = True
    SIMLI_API_KEY = os.getenv("SIMLI_API_KEY")
    SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID")
    if not SIMLI_API_KEY or not SIMLI_FACE_ID:
        logger.warning("SIMLI API Keys missing. Simli disabled.")
        SIMLI_SDK_AVAILABLE = False
except ImportError:
    logger.error("Simli SDK not found. Simli disabled.")
    SimliClient = None; SimliConfig = None; SIMLI_SDK_AVAILABLE = False; SIMLI_API_KEY = None; SIMLI_FACE_ID = None
except Exception as e:
    logger.exception(f"Simli import error: {e}")
    SimliClient = None; SimliConfig = None; SIMLI_SDK_AVAILABLE = False; SIMLI_API_KEY = None; SIMLI_FACE_ID = None

# --- Other Third-Party Imports ---
import gradio as gr
import numpy as np
import torch
from torch import nn
import av
import httpx
from PIL import Image
import soundfile as sf

try:
    from snac import SNAC
    logger.info("SNAC imported.")
except ImportError:
    logger.error("SNAC not found. pip install git+https://github.com/hubertsiuzdak/snac.git")
    SNAC = None; exit(1)

# --- Constants ---
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:1234")
TTS_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/completions"
TTS_MODEL = os.getenv("TTS_MODEL", "isaiahbjork/orpheus-3b-0.1-ft")
TTS_PROMPT_FORMAT = "<|audio|>{voice}: {text}<|eot_id|>"
TTS_PROMPT_STOP_TOKENS = ["<|eot_id|>", "<|audio|>"]
logger.info(f"TTS Server Endpoint: {TTS_API_ENDPOINT}, TTS Model: {TTS_MODEL}")
DEFAULT_TTS_TEMP = 0.8; DEFAULT_TTS_TOP_P = 0.9; DEFAULT_TTS_REP_PENALTY = 1.1
ORPHEUS_MIN_ID = 10; ORPHEUS_TOKENS_PER_LAYER = 4096; ORPHEUS_N_LAYERS = 7
ORPHEUS_MAX_ID = ORPHEUS_MIN_ID + (ORPHEUS_N_LAYERS * ORPHEUS_TOKENS_PER_LAYER)
SNAC_SAMPLE_RATE = 24000; SIMLI_TARGET_SAMPLE_RATE = 16000
TTS_STREAM_MIN_GROUPS = 40; TTS_STREAM_SILENCE_MS = 5
STREAM_TIMEOUT_SECONDS = 300
STREAM_HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
SSE_DATA_PREFIX = "data:"; SSE_DONE_MARKER = "[DONE]"
ALL_VOICES = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
DEFAULT_TTS_VOICE = ALL_VOICES[0]
TTS_FAILED_MSG = "(TTS generation failed or produced no audio)"
SIMLI_FAILED_MSG = "(Simli video generation failed)"
SIMLI_NO_FRAMES_MSG = "(Simli generated no video frames)"
SIMLI_SUCCESS_MSG = "(Simli Video Generated)"
FFMPEG_MISSING_MSG = "[ERROR: FFmpeg not found. Simli disabled.]"
SIMLI_UNAVAILABLE_MSG = "[ERROR: Simli SDK/Keys unavailable. Simli disabled.]"
NO_AUDIO_MSG = "[Error: Generate or Upload valid audio first.]"
INVALID_UPLOAD_MSG = "[Error: Uploaded file could not be copied or is invalid.]"
CONVERSION_FAILED_MSG = "[Error: Failed to convert uploaded audio to required format.]"
APP_TEMP_DIR = "gradio_temp_audio"

# --- Device Setup ---
tts_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"TTS Device: '{tts_device}'")

# --- FFmpeg Check ---
FFMPEG_PATH = "ffmpeg"
def check_ffmpeg():
    try:
        subprocess.run([FFMPEG_PATH, "-version"], capture_output=True, check=True, text=True)
        logger.info("ffmpeg found.")
        return True
    except FileNotFoundError:
        logger.error(f"'{FFMPEG_PATH}' not found.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"'{FFMPEG_PATH}' failed: {e.stderr}")
        return False
FFMPEG_AVAILABLE = check_ffmpeg()

# --- Helper Functions ---It is about the Filetering...
def parse_gguf_codes(response_text: str) -> List[int]:
    try:
        return [int(m) for m in re.findall(r"<custom_token_(\d+)>", response_text) if ORPHEUS_MIN_ID <= int(m) < ORPHEUS_MAX_ID]
    except Exception as e:
        logger.error(f"GGUF parse error: {e} on text: {response_text[:100]}")
        return []

def redistribute_codes(codes: List[int], model: nn.Module) -> Optional[np.ndarray]:
    if not codes or model is None: return None
    try:
        dev = next(model.parameters()).device; layers: List[List[int]] = [[], [], []]; groups = len(codes) // ORPHEUS_N_LAYERS;
        if groups == 0: return None
        valid = 0; codes_to_process = codes[:groups * ORPHEUS_N_LAYERS]
        for i in range(groups):
            idx=i*ORPHEUS_N_LAYERS; group=codes_to_process[idx:idx+ORPHEUS_N_LAYERS]; processed:List[Optional[int]]=[None]*ORPHEUS_N_LAYERS; ok=True
            for j, t_id in enumerate(group):
                if not (ORPHEUS_MIN_ID <= t_id < ORPHEUS_MAX_ID): ok = False; break
                layer_idx = (t_id - ORPHEUS_MIN_ID)//ORPHEUS_TOKENS_PER_LAYER; code_idx = (t_id - ORPHEUS_MIN_ID)%ORPHEUS_TOKENS_PER_LAYER
                if layer_idx != j: ok = False; break
                processed[j] = code_idx
            if ok:
                try:
                    if any(c is None for c in processed): continue
                    pg:List[int]=processed; layers[0].append(pg[0]); layers[1].append(pg[1]); layers[2].append(pg[2]); layers[2].append(pg[3]); layers[1].append(pg[4]); layers[2].append(pg[5]); layers[2].append(pg[6]); valid += 1
                except(IndexError, TypeError)as map_e: logger.error(f"Map error g{i}:{map_e}"); continue
        if valid==0: return None
        if len(layers[0])!=valid or len(layers[1])!=valid*2 or len(layers[2])!=valid*4: logger.error(f"Tensor len mismatch"); return None
        tensors=[torch.tensor(lc, device=dev, dtype=torch.long).unsqueeze(0) for lc in layers]
        with torch.no_grad(): audio=model.decode(tensors); torch.cuda.synchronize() if torch.cuda.is_available() else None
        audio_np=audio.detach().squeeze().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(audio_np)): logger.error("SNAC NaN/inf"); return np.zeros_like(audio_np)
        return audio_np
    except Exception as e: logger.exception(f"SNAC decode error: {e}"); return None

def apply_fade(audio_chunk: np.ndarray, sample_rate: int, fade_ms: int = 3) -> np.ndarray:
    num_fade_samples = int(sample_rate*(fade_ms/1000.0));
    if num_fade_samples <= 0 or audio_chunk.size < 2*num_fade_samples: return audio_chunk
    fade_in=np.linspace(0., 1., num_fade_samples, dtype=audio_chunk.dtype); fade_out=np.linspace(1., 0., num_fade_samples, dtype=audio_chunk.dtype)
    try: chunk_copy=audio_chunk.copy(); chunk_copy[:num_fade_samples]*=fade_in; chunk_copy[-num_fade_samples:]*=fade_out; return chunk_copy
    except ValueError as e: logger.warning(f"Fade failed: {e}"); return audio_chunk

async def convert_pcm_24k_to_16k(pcm_24k_bytes: bytes) -> Optional[bytes]:
    if not FFMPEG_AVAILABLE: logger.error("FFmpeg unavailable"); return None
    if not pcm_24k_bytes: return b''
    try:
        process=await asyncio.create_subprocess_exec( FFMPEG_PATH,'-f','s16le','-ar',str(SNAC_SAMPLE_RATE),'-ac','1','-i','pipe:0', '-f','s16le','-acodec','pcm_s16le','-ar',str(SIMLI_TARGET_SAMPLE_RATE),'-ac','1','-', stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout,stderr=await process.communicate(input=pcm_24k_bytes)
        if process.returncode!=0: logger.error(f"FFmpeg Error: {stderr.decode(errors='ignore')}"); return None
        return stdout
    except Exception as e: logger.exception(f"FFmpeg resample error: {e}"); return None

async def convert_any_audio_to_24k_mono_wav(input_path: str, output_path: str) -> bool:
    logger.info(f"Attempting FFmpeg conversion: {input_path} -> {output_path}")
    if not FFMPEG_AVAILABLE: logger.error("Cannot convert audio: FFmpeg not available."); return False
    try:
        process = await asyncio.create_subprocess_exec( FFMPEG_PATH, '-i', input_path, '-ar', str(SNAC_SAMPLE_RATE), '-ac', '1', '-acodec', 'pcm_s16le', '-y', output_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode(errors='ignore'); logger.error(f"FFmpeg conversion failed for {input_path}: {error_msg}")
            # --- Corrected Formatting ---
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass # Ignore error during cleanup of failed conversion
            # --- End Correction ---
            return False
        else:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 44:
                 logger.info(f"FFmpeg conversion successful: {output_path}")
                 return True
            else:
                logger.error(f"FFmpeg success, but output file {output_path} missing/empty.")
                # --- Corrected Formatting ---
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass # Ignore error during cleanup of empty file
                # --- End Correction ---
                return False
    except Exception as e:
        logger.exception(f"Error during FFmpeg conversion process for {input_path}: {e}")
        return False
# --- End Helper Functions ---

# --- Model Loading ---
logger.info("--- Loading SNAC Model ---")
snac_model: Optional[SNAC] = None
if SNAC is not None:
    try:
        with warnings.catch_warnings(): warnings.filterwarnings("ignore", category=FutureWarning, module="snac.snac")
        snac_model_instance = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        if snac_model_instance: snac_model = snac_model_instance.to(tts_device).eval(); logger.info(f"SNAC loaded to '{tts_device}'.")
        else: logger.error("SNAC.from_pretrained returned None.")
    except Exception as e: logger.exception("Fatal error loading SNAC model.")

# --- Simli Client Initialization ---
async def initialize_simli_client():
    if not SIMLI_SDK_AVAILABLE: logger.error("Simli SDK unavailable."); return None
    if not FFMPEG_AVAILABLE: logger.error("FFmpeg unavailable."); return None
    logger.info("Initializing Simli Client for new request...")
    config = SimliConfig( apiKey=SIMLI_API_KEY, faceId=SIMLI_FACE_ID, syncAudio=False, handleSilence=True, maxSessionLength=180, maxIdleTime=10)
    try: new_client = SimliClient(config); await new_client.Initialize(); logger.info("Simli Client Initialized."); return new_client
    except Exception as e: logger.exception(f"Failed to initialize Simli Client: {e}"); return None

# --- TTS Stream Generator ---
async def generate_speech_stream( text: str, voice: str, tts_temperature: float, tts_top_p: float, tts_repetition_penalty: float, buffer_groups_param: int, padding_ms_param: int,) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
    global snac_model;
    if not text.strip() or snac_model is None: return
    min_codes_required=max(1, buffer_groups_param)*ORPHEUS_N_LAYERS; silence_samples:int=int(SNAC_SAMPLE_RATE*(padding_ms_param/1000.0)) if padding_ms_param > 0 else 0
    payload={"model":TTS_MODEL, "prompt":TTS_PROMPT_FORMAT.format(voice=voice, text=text), "temperature":tts_temperature, "top_p":tts_top_p, "repeat_penalty":tts_repetition_penalty, "n_predict":-1, "stop":TTS_PROMPT_STOP_TOKENS, "stream":True }
    accumulated_codes:List[int]=[]; stream_successful=False
    try:
        async with httpx.AsyncClient(timeout=STREAM_TIMEOUT_SECONDS) as client:
            async with client.stream("POST", TTS_API_ENDPOINT, json=payload, headers=STREAM_HEADERS) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith(SSE_DATA_PREFIX): continue
                    json_str = line[len(SSE_DATA_PREFIX):].strip()
                    if json_str == SSE_DONE_MARKER: break
                    if not json_str: continue
                    try:
                        data = json.loads(json_str); chunk_text = ""
                        if "choices" in data and data["choices"]: chunk_text=data["choices"][0].get("delta", {}).get("content", "") or data["choices"][0].get("text", "")
                        if chunk_text:
                            new_codes = parse_gguf_codes(chunk_text)
                            if new_codes:
                                accumulated_codes.extend(new_codes)
                                if len(accumulated_codes) >= min_codes_required:
                                    num_groups=len(accumulated_codes)//ORPHEUS_N_LAYERS; codes_to_decode=accumulated_codes[:num_groups*ORPHEUS_N_LAYERS]; accumulated_codes=accumulated_codes[num_groups*ORPHEUS_N_LAYERS:]
                                    audio_chunk = await asyncio.to_thread(redistribute_codes, codes_to_decode, snac_model)
                                    if audio_chunk is not None and audio_chunk.size > 0:
                                        faded_chunk=apply_fade(audio_chunk, SNAC_SAMPLE_RATE, fade_ms=3)
                                        yield (SNAC_SAMPLE_RATE, np.concatenate((np.zeros(silence_samples, dtype=faded_chunk.dtype), faded_chunk, np.zeros(silence_samples, dtype=faded_chunk.dtype))) if silence_samples > 0 else faded_chunk)
                                        stream_successful = True
                        if "choices" in data and data["choices"] and data["choices"][0].get("finish_reason"): break
                    except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON: {json_str[:100]}..."); continue
                    except Exception as e: logger.exception(f"Chunk error: {json_str[:100]}..."); continue
        if len(accumulated_codes) >= ORPHEUS_N_LAYERS:
            num_groups=len(accumulated_codes)//ORPHEUS_N_LAYERS; codes_to_decode=accumulated_codes[:num_groups*ORPHEUS_N_LAYERS]
            audio_chunk = await asyncio.to_thread(redistribute_codes, codes_to_decode, snac_model)
            if audio_chunk is not None and audio_chunk.size > 0:
                faded_chunk=apply_fade(audio_chunk, SNAC_SAMPLE_RATE, fade_ms=3)
                yield (SNAC_SAMPLE_RATE, np.concatenate((np.zeros(silence_samples, dtype=faded_chunk.dtype), faded_chunk, np.zeros(silence_samples, dtype=faded_chunk.dtype))) if silence_samples > 0 else faded_chunk)
                stream_successful = True
    except Exception as e: logger.exception(f"❌ TTS Error: {e}")

# --- Simli Processing Tasks ---
async def resample_and_send_audio_task( client: SimliClient, input_24k_wav_path: str, output_16k_wav_path: str ):
    logger.info(f"--- Task: Resampling {input_24k_wav_path}, sending & saving to {output_16k_wav_path} ---"); sent_any_audio = False; buffer_size_samples = SNAC_SAMPLE_RATE // 4
    try:
        with sf.SoundFile(input_24k_wav_path, 'r') as infile:
            if infile.samplerate != SNAC_SAMPLE_RATE or infile.channels != 1: raise ValueError(f"Input WAV {input_24k_wav_path} must be {SNAC_SAMPLE_RATE}Hz Mono")
            with wave.open(output_16k_wav_path, 'wb') as outfile:
                outfile.setnchannels(1); outfile.setsampwidth(2); outfile.setframerate(SIMLI_TARGET_SAMPLE_RATE)
                while True:
                    audio_chunk_24k_f32 = infile.read(buffer_size_samples, dtype='float32')
                    if audio_chunk_24k_f32 is None or audio_chunk_24k_f32.shape[0] == 0: break
                    pcm16_24k = np.clip(audio_chunk_24k_f32 * 32767, -32768, 32767).astype(np.int16).tobytes()
                    pcm16_16k = await convert_pcm_24k_to_16k(pcm16_24k)
                    if pcm16_16k: await client.send(pcm16_16k); sent_any_audio = True; outfile.writeframes(pcm16_16k); await asyncio.sleep(0.01)
        if sent_any_audio: logger.info("Signaling end of speech."); await client.sendSilence(0.5)
        else: logger.warning("No audio chunks read/sent from file.")
    except Exception as e: logger.exception(f"Error during audio resampling/sending task for {input_24k_wav_path}."); raise
    logger.info(f"--- Task: Audio resample/send task finished ---")

async def receive_video_and_save_task(client: SimliClient, output_filepath: str, temp_audio_16k_path: str):
    logger.info("--- Task: Collecting video frames from Simli ---"); collected_original_frames=[]; frame_count=0; target_width=256; target_height=256
    try:
        logger.debug("Starting video stream iteration...")
        async for frame in client.getVideoStreamIterator(targetFormat='rgb24'):
            try: img=frame.to_ndarray(format="rgb24"); collected_original_frames.append(img); frame_count += 1
            except AttributeError: logger.error(f"Frame {frame_count} type {type(frame)} no 'to_ndarray'."); break
            except Exception as frame_e: logger.error(f"Error processing frame {frame_count}: {frame_e}"); continue
        logger.info(f"Finished collecting frames. Total: {frame_count}")
    except AttributeError as ae: logger.critical(f"getVideoStreamIterator not found:{ae}"); raise
    except Exception as e: logger.exception("Error receiving video frames.")

    if frame_count == 0:
        logger.warning("No frames received.")
        if os.path.exists(output_filepath):
            # --- Corrected Formatting ---
            try:
                os.remove(output_filepath)
                logger.debug(f"Removed potentially empty file: {output_filepath}")
            except OSError as e:
                logger.warning(f"Failed to remove file {output_filepath}: {e}")
            # --- End Correction ---
        return

    logger.info(f"Muxing {frame_count} frames -> {output_filepath} ({target_width}x{target_height}) with {temp_audio_16k_path}"); mux_success=False
    container = None # Define container outside try for finally
    try:
        if not os.path.exists(temp_audio_16k_path) or os.path.getsize(temp_audio_16k_path) <= 44:
            raise FileNotFoundError(f"16kHz audio missing:{temp_audio_16k_path}")
        container = av.open(output_filepath, mode='w')
        fps = 30
        vstream = container.add_stream('libx264', rate=fps)
        vstream.width = target_width
        vstream.height = target_height
        vstream.pix_fmt = 'yuv420p'
        # Add Audio Stream
        try:
            with wave.open(temp_audio_16k_path, 'rb') as wav:
                if wav.getframerate() != SIMLI_TARGET_SAMPLE_RATE or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError("Incorrect 16kHz audio format")
                astream = container.add_stream('aac', rate=SIMLI_TARGET_SAMPLE_RATE, layout='mono')
                frame_size = 1024
                while True:
                    chunk_bytes = wav.readframes(frame_size)
                    if not chunk_bytes: break
                    samples = np.frombuffer(chunk_bytes, dtype=np.int16)
                    if samples.size == 0: continue
                    afr = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format='s16', layout='mono')
                    afr.sample_rate = SIMLI_TARGET_SAMPLE_RATE
                    afr.pts = None
                    for packet in astream.encode(afr): container.mux(packet)
                for packet in astream.encode(None): container.mux(packet) # Flush audio
        except Exception as audio_mux_e:
             logger.exception(f"Audio mux error:{audio_mux_e}")
             # Don't raise here, allow video muxing, but mark mux as potentially incomplete
             # Note: container might need closing if audio add_stream failed early
             if container: container.close() # Attempt close if container exists
             raise # Re-raise to signal failure

        logger.debug("Resizing/encoding video frames...")
        for i, original_img in enumerate(collected_original_frames):
            try:
                pil_img = Image.fromarray(original_img)
                resized_pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                resized_img_np = np.array(resized_pil_img)
                frame = av.VideoFrame.from_ndarray(resized_img_np, format='rgb24')
                frame.pts = i
                for packet in vstream.encode(frame): container.mux(packet)
            except Exception as encode_err:
                logger.error(f"Error resize/encode frame {i}:{encode_err}")
                continue # Skip faulty frame

        for packet in vstream.encode(None): container.mux(packet) # Flush video
        container.close() # Close after successful muxing
        logger.info(f"Muxed {frame_count} frames -> {output_filepath}")
        mux_success = True

    except Exception as mux_e:
        logger.exception(f"Video muxing error:{mux_e}")
        if container: # Ensure container is closed if opened
             try: container.close()
             except Exception: pass # Ignore errors closing already errored container
        # --- Corrected Formatting ---
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
                logger.debug(f"Removed partially muxed file:{output_filepath}")
            except OSError as e:
                logger.warning(f"Failed remove partial file {output_filepath}:{e}")
        # --- End Correction ---
        raise # Re-raise muxing error
    finally:
        # --- Corrected Formatting ---
        # Clean up temp 16k audio file used for muxing
        if os.path.exists(temp_audio_16k_path):
            try:
                os.remove(temp_audio_16k_path)
                logger.debug(f"Cleaned temp 16k audio: {temp_audio_16k_path}")
            except OSError as e:
                logger.warning(f"Failed clean temp 16k audio {temp_audio_16k_path}:{e}")
        # --- End Correction ---

# --- Gradio Processing Functions ---
async def process_tts_to_audio( text: str, voice: str, temp: float, top_p: float, rep_pen: float, buffer: int, padding: int ) -> AsyncGenerator[Dict[gr.component, Any], None]:
    global tts_audio_output, status_textbox, audio_filepath_state, generate_audio_button, generate_simli_button, upload_audio_button, uploaded_audio_preview, uploaded_filepath_state

    if not text or not text.strip():
        yield { status_textbox: "Error: No text provided.", generate_audio_button: gr.update(interactive=True) }
        return
    if snac_model is None:
        yield { status_textbox: "[Error: SNAC model not loaded]", generate_audio_button: gr.update(interactive=True) }
        return

    yield {
        status_textbox: "Generating audio...", tts_audio_output: None, audio_filepath_state: None,
        uploaded_audio_preview: None, uploaded_filepath_state: None, # Clear upload preview/state too
        generate_audio_button: gr.update(interactive=False),
        generate_simli_button: gr.update(interactive=False),
        upload_audio_button: gr.update(interactive=False)
    }

    os_temp_wav_path = None
    app_temp_wav_path = None
    audio_generated = False
    try:
        with tempfile.NamedTemporaryFile(suffix="_24k.wav", delete=False) as tmp_wav:
            os_temp_wav_path = tmp_wav.name
        logger.info(f"Generating 24kHz audio to OS temp: {os_temp_wav_path}")

        with sf.SoundFile(os_temp_wav_path, mode='w', samplerate=SNAC_SAMPLE_RATE, channels=1, subtype='PCM_16') as outfile:
             async for sr, audio_chunk in generate_speech_stream( text, voice, temp, top_p, rep_pen, buffer, padding ):
                if sr != SNAC_SAMPLE_RATE: continue
                if audio_chunk is not None and audio_chunk.size > 0: outfile.write(audio_chunk); audio_generated = True

        if audio_generated and os.path.exists(os_temp_wav_path) and os.path.getsize(os_temp_wav_path) > 44:
            logger.info(f"Audio data written to {os_temp_wav_path}")
            os.makedirs(APP_TEMP_DIR, exist_ok=True)
            unique_filename = f"tts_output_{uuid.uuid4()}.wav"
            app_temp_wav_path = os.path.join(APP_TEMP_DIR, unique_filename)
            shutil.copy2(os_temp_wav_path, app_temp_wav_path) # Copy to serve dir
            logger.info(f"Copied audio to Gradio path: {app_temp_wav_path}")
            yield {
                status_textbox: "Audio generated successfully.",
                tts_audio_output: gr.update(value=app_temp_wav_path), # Update component value
                audio_filepath_state: app_temp_wav_path, # Store the *copied* path
            }
        else:
            logger.error("TTS finished but no valid audio data produced."); audio_generated = False
            yield { status_textbox: TTS_FAILED_MSG, tts_audio_output: None, audio_filepath_state: None }

    except Exception as e:
        logger.exception("Error during TTS audio generation."); audio_generated = False
        yield { status_textbox: f"[TTS Error: {e}]", tts_audio_output: None, audio_filepath_state: None }

    finally:
        if os_temp_wav_path and os.path.exists(os_temp_wav_path):
            # --- Corrected Formatting ---
            try:
                os.remove(os_temp_wav_path)
                logger.debug(f"Cleaned OS temp file: {os_temp_wav_path}")
            except OSError as e:
                logger.warning(f"Failed clean OS temp file {os_temp_wav_path}: {e}")
            # --- End Correction ---

        # Always re-enable buttons
        yield {
             generate_audio_button: gr.update(interactive=True),
             generate_simli_button: gr.update(interactive=audio_generated), # Enable Simli only if audio generated
             upload_audio_button: gr.update(interactive=True)
        }
    logger.info("--- TTS to Audio Request Finished ---")


async def handle_audio_upload(uploaded_file_path: Optional[str]) -> Dict[gr.component, Any]:
    global uploaded_audio_preview, uploaded_filepath_state, upload_audio_button, status_textbox, generate_simli_button, tts_audio_output, audio_filepath_state

    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        logger.warning("handle_audio_upload called with invalid path.")
        return { status_textbox: "[Error: Upload failed or file not found.]", uploaded_audio_preview: None, uploaded_filepath_state: None, upload_audio_button: gr.update(interactive=False) }

    app_temp_upload_path = None
    upload_successful = False
    try:
        os.makedirs(APP_TEMP_DIR, exist_ok=True)
        original_filename = os.path.basename(uploaded_file_path)
        filename_base, filename_ext = os.path.splitext(original_filename)
        # Ensure extension is .wav or .mp3 or similar for better browser compatibility
        safe_ext = filename_ext.lower() if filename_ext.lower() in ['.wav', '.mp3', '.ogg', '.flac'] else '.wav'
        unique_filename = f"{filename_base}_{uuid.uuid4()}{safe_ext}"
        app_temp_upload_path = os.path.join(APP_TEMP_DIR, unique_filename)

        logger.info(f"Copying uploaded file '{uploaded_file_path}' to '{app_temp_upload_path}'")
        shutil.copy2(uploaded_file_path, app_temp_upload_path)

        if os.path.exists(app_temp_upload_path):
            logger.info("Upload copy successful.")
            upload_successful = True
            return {
                status_textbox: f"Uploaded '{original_filename}'. Ready for Simli.",
                uploaded_audio_preview: gr.update(value=app_temp_upload_path),
                uploaded_filepath_state: app_temp_upload_path,
                upload_audio_button: gr.update(interactive=True), # Enable the button for this upload
                tts_audio_output: None, # Clear TTS display
                audio_filepath_state: None, # Clear TTS state
                generate_simli_button: gr.update(interactive=False) # Disable TTS->Simli button
            }
        else:
            logger.error("File copy seemed to succeed but destination file not found.")
            return { status_textbox: "[Error: Failed to copy uploaded file.]", uploaded_audio_preview: None, uploaded_filepath_state: None, upload_audio_button: gr.update(interactive=False) }

    except Exception as e:
        logger.exception(f"Error copying uploaded file {uploaded_file_path}: {e}")
        if app_temp_upload_path and os.path.exists(app_temp_upload_path):
            # --- Corrected Formatting ---
            try:
                os.remove(app_temp_upload_path)
            except OSError:
                pass # Ignore error removing partial copy
            # --- End Correction ---
        return { status_textbox: f"[Error: Could not process upload: {e}]", uploaded_audio_preview: None, uploaded_filepath_state: None, upload_audio_button: gr.update(interactive=False) }


async def process_audio_to_simli(
    input_audio_path: Optional[str]
) -> AsyncGenerator[Dict[gr.component, Any], None]:
    global output_video, status_textbox, generate_audio_button, generate_simli_button, upload_audio_button

    if not input_audio_path or not os.path.exists(input_audio_path):
        yield { status_textbox: NO_AUDIO_MSG, output_video: None }
        # Let finally handle button re-enabling
        return
    logger.info(f"process_audio_to_simli received path: {input_audio_path}")

    if not SIMLI_SDK_AVAILABLE: yield { status_textbox: SIMLI_UNAVAILABLE_MSG, output_video: None }; return
    if not FFMPEG_AVAILABLE: yield { status_textbox: FFMPEG_MISSING_MSG, output_video: None }; return

    logger.info(f"--- Starting Audio to Simli Request (Input: {input_audio_path}) ---")
    yield {
        output_video: None, status_textbox: "Preparing audio...",
        generate_audio_button: gr.update(interactive=False),
        generate_simli_button: gr.update(interactive=False),
        upload_audio_button: gr.update(interactive=False)
    }

    converted_24k_mono_wav_path = None
    output_video_path = None
    temp_16k_audio_path = None
    simli_client_instance = None
    simli_success = False
    final_status_message = "Starting..."

    try:
        # --- Conversion Step ---
        with tempfile.NamedTemporaryFile(suffix="_24k_mono_converted.wav", delete=False) as tmp_conv:
            converted_24k_mono_wav_path = tmp_conv.name

        conversion_success = await convert_any_audio_to_24k_mono_wav(input_audio_path, converted_24k_mono_wav_path)

        if not conversion_success:
            yield { status_textbox: CONVERSION_FAILED_MSG, output_video: None }
            # Clean up failed conversion file
            if converted_24k_mono_wav_path and os.path.exists(converted_24k_mono_wav_path):
                # --- Corrected Formatting ---
                try:
                    os.remove(converted_24k_mono_wav_path)
                except OSError:
                    pass
                # --- End Correction ---
            return # Allow finally block to re-enable buttons
        # --- End Conversion Step ---

        # --- Proceed with Simli using the CONVERTED audio path ---
        yield {status_textbox: "Initializing Simli..."}
        simli_client_instance = await initialize_simli_client()
        if not simli_client_instance:
            yield { status_textbox: "[Error: Simli Init Fail]"}
            return # Allow finally block

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video, \
             tempfile.NamedTemporaryFile(suffix="_16k.wav", delete=False) as tmp_audio_16k:
            output_video_path = tmp_video.name
            temp_16k_audio_path = tmp_audio_16k.name
        logger.info(f"Temp files: Vid={output_video_path}, Aud16k={temp_16k_audio_path}")

        yield {status_textbox: "Resampling audio, sending & receiving video..."}
        send_task = asyncio.create_task(resample_and_send_audio_task(simli_client_instance, converted_24k_mono_wav_path, temp_16k_audio_path))
        receive_task = asyncio.create_task(receive_video_and_save_task(simli_client_instance, output_video_path, temp_16k_audio_path))

        logger.info("Waiting for Simli Tasks...")
        results = await asyncio.gather(send_task, receive_task, return_exceptions=True)
        send_result, receive_result = results
        receive_error = isinstance(receive_result, Exception)
        send_error = isinstance(send_result, Exception)

        # Determine final status
        if receive_error:
            logger.error(f"Receive/Mux task failed:{receive_result}"); final_status_message=f"{SIMLI_FAILED_MSG} (Receive/Mux Error)"; simli_success=False
            if output_video_path and os.path.exists(output_video_path):
                # --- Corrected Formatting ---
                try:
                    os.remove(output_video_path)
                except OSError as e:
                    logger.warning(f"Failed remove broken video {output_video_path}:{e}")
                # --- End Correction ---
            output_video_path=None
        elif send_error:
            logger.error(f"Audio resample/send task failed:{send_result}")
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 100: final_status_message=f"{SIMLI_SUCCESS_MSG} (Warn: Audio Send Fail)"; simli_success=True
            else: final_status_message=f"{SIMLI_FAILED_MSG} (Audio Send Error)"; simli_success=False; output_video_path=None
        else:
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 100: final_status_message=SIMLI_SUCCESS_MSG; simli_success=True
            else: logger.error("Tasks OK, but output video missing/empty."); final_status_message=SIMLI_NO_FRAMES_MSG; simli_success=False; output_video_path=None

        yield {status_textbox: final_status_message}

    except Exception as e:
        logger.exception("Error in Audio to Simli processing loop.")
        final_status_message = f"[Runtime Error: {e}]"; yield { status_textbox: final_status_message, output_video: None }; simli_success = False
        if output_video_path and os.path.exists(output_video_path):
            # --- Corrected Formatting ---
            try:
                os.remove(output_video_path)
            except OSError as e_rem:
                logger.warning(f"Failed remove video file on error {output_video_path}:{e_rem}")
            # --- End Correction ---

    finally:
        # Clean up Simli client
        if simli_client_instance:
             try:
                 logger.info("Closing Simli client instance (audio->simli finally)...")
                 if hasattr(simli_client_instance, "close") and callable(getattr(simli_client_instance, "close")):
                      await simli_client_instance.close(); logger.info("Simli client closed via finally.")
                 else: logger.warning("Audio->Simli finally: Client object has no callable 'close' method.")
             except Exception as close_e: logger.error(f"Error closing Simli client (audio->simli finally):{close_e}")

        # Clean up the CONVERTED 24k file
        if converted_24k_mono_wav_path and os.path.exists(converted_24k_mono_wav_path):
            # --- Corrected Formatting ---
            try:
                os.remove(converted_24k_mono_wav_path)
                logger.debug(f"Cleaned temp CONVERTED 24k audio: {converted_24k_mono_wav_path}")
            except OSError as e:
                logger.warning(f"Failed clean temp CONVERTED 24k audio {converted_24k_mono_wav_path}: {e}")
            # --- End Correction ---

        # Clean up intermediate 16k audio file (already handled in receive_video_and_save_task finally)

        # Re-enable all action buttons
        yield {
            generate_audio_button: gr.update(interactive=True),
            generate_simli_button: gr.update(interactive=True),
            upload_audio_button: gr.update(interactive=True)
        }

    # Final yield for video
    if simli_success and output_video_path:
        logger.info(f"Yielding final video path: {output_video_path}"); yield {output_video:gr.update(value=output_video_path)}
    else:
         logger.warning("Clearing video output (failed or no path)."); yield {output_video:None}
    logger.info("--- Audio to Simli Request Finished ---")
# --- End Gradio Processing Functions ---


# --- Gradio UI Definition ---
css_rules = """
#simli_video_output video { width: 100% !important; height: 100% !important; object-fit: contain; }
#simli_video_output { max-width: 256px !important; max-height: 256px !important; width: 256px !important; height: 256px !important; overflow: hidden; margin: auto; }
#tts_audio_output, #uploaded_audio_preview { margin-top: 10px; }
#upload_audio { margin-top: 10px; }
"""

with gr.Blocks(css=css_rules, title="TTS/Upload -> Simli") as demo:
    gr.Markdown("# TTS/Upload Audio -> Simli Avatar Meme Factory")
    gr.Markdown("Generate TTS audio OR Upload an audio file (WAV, MP3, etc.), then generate the Simli video.")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Option 1: Generate TTS Audio")
            text_input = gr.Textbox(label="Text to Speak", lines=4, placeholder="Enter text here...")
            tts_voice_dd = gr.Dropdown(choices=ALL_VOICES, value=DEFAULT_TTS_VOICE, label="TTS Voice")
            with gr.Accordion("TTS Parameters", open=False):
                tts_temp_slider = gr.Slider(0.1, 2.0, value=DEFAULT_TTS_TEMP, step=0.05, label="Temperature")
                tts_top_p_slider = gr.Slider(0.1, 1.0, value=DEFAULT_TTS_TOP_P, step=0.05, label="Top P")
                tts_rep_penalty_slider = gr.Slider(1.0, 1.5, value=DEFAULT_TTS_REP_PENALTY, step=0.01, label="Repetition Penalty")
                tts_buffer_groups_slider = gr.Slider(minimum=1, maximum=100, value=TTS_STREAM_MIN_GROUPS, step=1, label="TTS Buffer (Groups)")
                tts_padding_ms_slider = gr.Slider(minimum=0, maximum=500, value=TTS_STREAM_SILENCE_MS, step=1, label="TTS Padding (ms)")
            generate_audio_button = gr.Button("1a. Generate Audio", variant="secondary")
            tts_audio_output = gr.Audio(label="Generated TTS Audio Output", type="filepath", elem_id="tts_audio_output")
            audio_filepath_state = gr.State(None)

            gr.Markdown("---")

            gr.Markdown("### Option 2: Upload Audio File")
            upload_audio_input = gr.File(label="Upload Audio (WAV, MP3, etc.)", type="filepath", file_types=["audio"], elem_id="upload_audio")
            uploaded_audio_preview = gr.Audio(label="Uploaded Audio Preview", type="filepath", elem_id="uploaded_audio_preview")
            uploaded_filepath_state = gr.State(None)
            upload_audio_button = gr.Button("2b. Generate Simli from Upload", variant="secondary", interactive=False) # Initially disabled

        with gr.Column(scale=3):
            gr.Markdown("### Simli Video Output")
            generate_simli_button = gr.Button("2a. Generate Simli from Generated Audio", variant="primary", interactive=False)
            output_video = gr.Video( label="Simli Avatar Output (256x256)", interactive=False, elem_id="simli_video_output")
            status_textbox = gr.Textbox(label="Status", value="Idle", interactive=False, lines=3)

    # --- Connect UI Elements ---
    generate_audio_button.click(
        fn=process_tts_to_audio,
        inputs=[ text_input, tts_voice_dd, tts_temp_slider, tts_top_p_slider, tts_rep_penalty_slider, tts_buffer_groups_slider, tts_padding_ms_slider ],
        outputs=[tts_audio_output, status_textbox, audio_filepath_state, generate_audio_button, generate_simli_button, upload_audio_button, uploaded_audio_preview, uploaded_filepath_state]
    )
    generate_simli_button.click(
        fn=process_audio_to_simli,
        inputs=[audio_filepath_state],
        outputs=[output_video, status_textbox, generate_audio_button, generate_simli_button, upload_audio_button]
    )
    # Handle File Upload - Update preview, state, and buttons
    upload_audio_input.upload(
        fn=handle_audio_upload,
        inputs=[upload_audio_input],
        outputs=[uploaded_audio_preview, uploaded_filepath_state, upload_audio_button, status_textbox, tts_audio_output, audio_filepath_state, generate_simli_button]
    )
    # Generate Simli from Upload button
    upload_audio_button.click(
        fn=process_audio_to_simli,
        inputs=[uploaded_filepath_state], # Use the state holding the copied path
        outputs=[output_video, status_textbox, generate_audio_button, generate_simli_button, upload_audio_button]
    )
# --- End Gradio UI ---

# --- Launch Gradio App ---
if __name__ == "__main__":
    # Dependency checks
    try: import PIL; logger.info("Pillow found.")
    except ImportError: logger.critical("Pillow not found. pip install Pillow. Exiting."); exit(1)
    try: import soundfile; logger.info("Soundfile found.")
    except ImportError: logger.critical("Soundfile not found. pip install SoundFile. Exiting."); exit(1)
    if not FFMPEG_AVAILABLE: logger.critical("FFmpeg missing.")
    if not SIMLI_SDK_AVAILABLE: logger.critical("Simli SDK missing/unavailable.")
    if snac_model is None: logger.error("SNAC Model not loaded.")

    # Launch only if core dependencies are met
    if snac_model is not None and FFMPEG_AVAILABLE:
        os.makedirs(APP_TEMP_DIR, exist_ok=True)
        logger.info(f"Created/Ensured app temporary directory: {APP_TEMP_DIR}")
        logger.info("Launching TTS/Upload -> Simli Gradio Interface...")
        demo.launch(share=False)
        logger.info("Gradio Interface Closed.")
    else:
        logger.critical("Exiting because SNAC model or FFmpeg is missing/failed to load.")
# --- End Launch ---