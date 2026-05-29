"""
Video Nodes for ComfyUI.
"""
import os
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from fractions import Fraction

from .base import *


def _normalize_frame_batch(frames: Tensor) -> Tensor:
	if frames.dim() == 3:
		return frames.unsqueeze(0)
	return frames


def _frame_batch_to_cv2_list(frames: Tensor) -> List[np.ndarray]:
	frames = _normalize_frame_batch(frames)
	frames_np = frames.cpu().numpy()
	if frames_np.max() <= 1.0:
		frames_np = frames_np * 255
	frames_np = frames_np.clip(0, 255).astype(np.uint8)
	return [frame[:, :, ::-1].copy() for frame in frames_np]


def _sanitize_path_part(value: str) -> str:
	safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value.strip())
	return safe.strip("._") or "video"


def _resolve_output_directory() -> str:
	try:
		import folder_paths
		output_dir = folder_paths.get_output_directory()
	except Exception:
		output_dir = os.path.join(os.getcwd(), "output")
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def _make_output_video_path(filename_prefix: str = "FaceFusion/video") -> str:
	prefix_parts = [
		_sanitize_path_part(part)
		for part in filename_prefix.replace("\\", "/").split("/")
		if part.strip()
	]
	if not prefix_parts:
		prefix_parts = ["FaceFusion", "video"]

	output_dir = _resolve_output_directory()
	target_dir = os.path.join(output_dir, *prefix_parts[:-1])
	os.makedirs(target_dir, exist_ok=True)
	stem = prefix_parts[-1]
	suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
	return os.path.join(target_dir, f"{stem}_{suffix}.mp4")


def _sample_frame_indices(frame_count: int) -> List[int]:
	if frame_count <= 0:
		return []
	return sorted(set([0, frame_count // 2, frame_count - 1]))


def _find_ffmpeg_path() -> Optional[str]:
	ffmpeg_path = shutil.which("ffmpeg")
	if ffmpeg_path:
		return ffmpeg_path
	try:
		import imageio_ffmpeg
		return imageio_ffmpeg.get_ffmpeg_exe()
	except Exception:
		return None


def _find_ffprobe_path(ffmpeg_path: Optional[str] = None) -> Optional[str]:
	ffprobe_path = shutil.which("ffprobe")
	if ffprobe_path:
		return ffprobe_path
	if ffmpeg_path:
		for name in ("ffprobe.exe", "ffprobe"):
			candidate = os.path.join(os.path.dirname(ffmpeg_path), name)
			if os.path.exists(candidate):
				return candidate
	return None


def _parse_frame_rate(value: Optional[str], fallback: float = 24.0) -> float:
	if not value or value == "0/0":
		return fallback
	try:
		return float(Fraction(value))
	except Exception:
		try:
			return float(value)
		except Exception:
			return fallback


def _probe_video(video_path: str, fallback_frame_rate: float = 24.0) -> Dict[str, Any]:
	ffmpeg_path = _find_ffmpeg_path()
	ffprobe_path = _find_ffprobe_path(ffmpeg_path)
	if not ffprobe_path:
		raise RuntimeError("ffprobe not found; video_path input requires ffmpeg/ffprobe")

	command = [
		ffprobe_path,
		"-v",
		"error",
		"-select_streams",
		"v:0",
		"-show_entries",
		"stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
		"-of",
		"json",
		video_path,
	]
	result = subprocess.run(command, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError(f"ffprobe failed for video_path: {result.stderr.strip()}")

	data = json.loads(result.stdout or "{}")
	streams = data.get("streams") or []
	if not streams:
		raise RuntimeError(f"No video stream found in: {video_path}")

	stream = streams[0]
	width = int(stream.get("width") or 0)
	height = int(stream.get("height") or 0)
	if width <= 0 or height <= 0:
		raise RuntimeError(f"Invalid video dimensions for: {video_path}")

	fps = _parse_frame_rate(stream.get("avg_frame_rate"), fallback_frame_rate)
	if fps <= 0:
		fps = _parse_frame_rate(stream.get("r_frame_rate"), fallback_frame_rate)

	frame_count = 0
	try:
		frame_count = int(stream.get("nb_frames") or 0)
	except Exception:
		frame_count = 0
	duration = 0.0
	try:
		duration = float(stream.get("duration") or 0.0)
	except Exception:
		duration = 0.0
	if frame_count <= 0 and duration > 0 and fps > 0:
		frame_count = int(round(duration * fps))

	return {
		"width": width,
		"height": height,
		"fps": fps,
		"frame_count": frame_count,
		"duration": duration,
	}


def _read_exact(pipe, size: int) -> bytes:
	chunks = []
	remaining = size
	while remaining > 0:
		chunk = pipe.read(remaining)
		if not chunk:
			break
		chunks.append(chunk)
		remaining -= len(chunk)
	return b"".join(chunks)


def _iter_video_path_frames(video_path: str, width: int, height: int):
	ffmpeg_path = _find_ffmpeg_path()
	if not ffmpeg_path:
		raise RuntimeError("ffmpeg not found; video_path input requires ffmpeg")

	command = [
		ffmpeg_path,
		"-v",
		"error",
		"-i",
		video_path,
		"-map",
		"0:v:0",
		"-f",
		"rawvideo",
		"-pix_fmt",
		"bgr24",
		"pipe:1",
	]
	process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	frame_size = width * height * 3
	try:
		while True:
			frame_bytes = _read_exact(process.stdout, frame_size)
			if not frame_bytes:
				break
			if len(frame_bytes) != frame_size:
				raise RuntimeError("Incomplete frame read from ffmpeg")
			yield np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()
	finally:
		if process.stdout:
			process.stdout.close()
		stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
		return_code = process.wait()
		if return_code != 0:
			raise RuntimeError(f"ffmpeg decode failed: {stderr.strip()}")


def _read_video_frame_at(video_path: str, seconds: float, width: int, height: int) -> Optional[np.ndarray]:
	ffmpeg_path = _find_ffmpeg_path()
	if not ffmpeg_path:
		return None

	command = [
		ffmpeg_path,
		"-v",
		"error",
		"-ss",
		f"{max(0.0, seconds):.6f}",
		"-i",
		video_path,
		"-frames:v",
		"1",
		"-f",
		"rawvideo",
		"-pix_fmt",
		"bgr24",
		"pipe:1",
	]
	result = subprocess.run(command, capture_output=True)
	frame_size = width * height * 3
	if result.returncode != 0 or len(result.stdout) < frame_size:
		return None
	return np.frombuffer(result.stdout[:frame_size], dtype=np.uint8).reshape((height, width, 3)).copy()


def _sample_video_path_frames(video_path: str, metadata: Dict[str, Any]) -> List[np.ndarray]:
	width = metadata["width"]
	height = metadata["height"]
	fps = metadata.get("fps") or 24.0
	frame_count = metadata.get("frame_count") or 0
	duration = metadata.get("duration") or 0.0

	if frame_count > 0:
		times = [idx / fps for idx in _sample_frame_indices(frame_count)]
	elif duration > 0:
		times = [0.0, duration * 0.5, max(0.0, duration - (1.0 / max(fps, 1.0)))]
	else:
		times = [0.0]

	samples = []
	for seconds in sorted(set(round(time_value, 6) for time_value in times)):
		frame = _read_video_frame_at(video_path, seconds, width, height)
		if frame is not None:
			samples.append(frame)
	return samples


def _clean_video_path(video_path: Optional[str]) -> str:
	return (video_path or "").strip().strip("\"'")


def _prepare_target_video_source(
	target_frames: Optional[Tensor],
	video_path: str,
	frame_rate: float,
	target_audio: Optional[Dict]
) -> Dict[str, Any]:
	input_path = _clean_video_path(video_path)
	if input_path:
		if not os.path.isfile(input_path):
			raise FileNotFoundError(f"video_path not found: {input_path}")
		metadata = _probe_video(input_path, frame_rate)
		width = metadata["width"]
		height = metadata["height"]
		effective_frame_rate = metadata.get("fps") or frame_rate
		return {
			"is_path": True,
			"input_path": input_path,
			"frame_iter_factory": lambda: _iter_video_path_frames(input_path, width, height),
			"sample_frames": _sample_video_path_frames(input_path, metadata),
			"frame_rate": effective_frame_rate,
			"width": width,
			"height": height,
			"audio_source_path": "" if target_audio is not None else input_path,
		}

	if target_frames is None:
		raise ValueError("Provide either target_frames or video_path")

	frame_list = _frame_batch_to_cv2_list(target_frames)
	if not frame_list:
		raise ValueError("No frames provided")
	height, width = frame_list[0].shape[:2]
	return {
		"is_path": False,
		"input_path": "",
		"frame_iter_factory": lambda: iter(frame_list),
		"sample_frames": [frame_list[idx] for idx in _sample_frame_indices(len(frame_list))],
		"frame_rate": frame_rate,
		"width": width,
		"height": height,
		"audio_source_path": "",
	}


def _write_original_video_source(
	video_source: Dict[str, Any],
	target_frames: Optional[Tensor],
	output_path: str,
	frame_rate: float,
	target_audio: Optional[Dict]
) -> None:
	if video_source.get("is_path") and video_source.get("input_path"):
		shutil.copyfile(video_source["input_path"], output_path)
		return
	if target_frames is None:
		raise ValueError("No target_frames available for fallback output")
	_write_tensor_video_output(target_frames, output_path, frame_rate, target_audio)


def _write_audio_temp_file(audio: Dict) -> Optional[str]:
	try:
		import torchaudio
		waveform = audio["waveform"]
		sample_rate = audio["sample_rate"]
		if waveform.dim() == 3:
			waveform = waveform.squeeze(0)
		with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
			audio_path = temp_audio.name
		torchaudio.save(audio_path, waveform, sample_rate, format="wav")
		return audio_path
	except Exception as exc:
		print(f"[FaceFusionVideo] Failed to prepare audio for output path: {exc}")
		return None


def _open_ffmpeg_encoder(
	output_path: str,
	width: int,
	height: int,
	frame_rate: float,
	audio: Optional[Dict] = None,
	audio_source_path: str = ""
):
	ffmpeg_path = _find_ffmpeg_path()
	if not ffmpeg_path:
		raise RuntimeError("ffmpeg not found; video output requires ffmpeg")

	audio_path = _write_audio_temp_file(audio) if audio is not None else None
	command = [
		ffmpeg_path,
		"-y",
		"-v",
		"error",
		"-f",
		"rawvideo",
		"-pix_fmt",
		"bgr24",
		"-s",
		f"{width}x{height}",
		"-r",
		str(frame_rate),
		"-i",
		"pipe:0",
	]
	if audio_path:
		command.extend(["-i", audio_path])
	elif audio_source_path:
		command.extend(["-i", audio_source_path])
	command.extend(["-map", "0:v:0"])
	if audio_path:
		command.extend(["-map", "1:a:0"])
	elif audio_source_path:
		command.extend(["-map", "1:a:0?"])
	command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
	if audio_path or audio_source_path:
		command.extend(["-c:a", "aac", "-shortest"])
	else:
		command.extend(["-an"])
	command.append(output_path)

	process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
	return process, audio_path


def _close_ffmpeg_encoder(process, audio_path: Optional[str]) -> None:
	try:
		if process.stdin:
			try:
				process.stdin.close()
			except BrokenPipeError:
				pass
		stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
		return_code = process.wait()
		if return_code != 0:
			raise RuntimeError(f"ffmpeg encode failed: {stderr.strip()}")
	finally:
		if audio_path and os.path.exists(audio_path):
			os.unlink(audio_path)


def _write_tensor_video_output(frames: Tensor, output_path: str, frame_rate: float, audio: Optional[Dict] = None) -> None:
	_write_bgr_video_output(_frame_batch_to_cv2_list(frames), output_path, frame_rate, audio)


def _write_bgr_video_output(
	frames: List[np.ndarray],
	output_path: str,
	frame_rate: float,
	audio: Optional[Dict] = None,
	audio_source_path: str = ""
) -> None:
	if not frames:
		raise ValueError("No frames provided")

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	height, width = frames[0].shape[:2]
	process, audio_path = _open_ffmpeg_encoder(output_path, width, height, frame_rate, audio, audio_source_path)
	try:
		for frame in frames:
			process.stdin.write(frame[:, :, :3].tobytes())
	finally:
		_close_ffmpeg_encoder(process, audio_path)


def _write_processed_video_output(
	frame_iter,
	swap_face_fn,
	chunk_size: int,
	max_workers: int,
	frame_rate: float,
	output_path: str,
	audio: Optional[Dict] = None,
	audio_source_path: str = "",
	width: Optional[int] = None,
	height: Optional[int] = None
) -> None:
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	process = None
	audio_path = None
	pending_frames = []
	written = 0

	def flush_pending(frames: List[np.ndarray]) -> None:
		nonlocal process, audio_path, written, width, height
		if not frames:
			return
		worker_count = max(1, min(max_workers, len(frames)))
		with ThreadPoolExecutor(max_workers=worker_count) as executor:
			for frame_cv2 in executor.map(swap_face_fn, frames):
				if process is None:
					frame_height, frame_width = frame_cv2.shape[:2]
					width = width or frame_width
					height = height or frame_height
					process, audio_path = _open_ffmpeg_encoder(output_path, width, height, frame_rate, audio, audio_source_path)
				process.stdin.write(frame_cv2[:, :, :3].tobytes())
				written += 1

	try:
		if width is not None and height is not None:
			process, audio_path = _open_ffmpeg_encoder(output_path, width, height, frame_rate, audio, audio_source_path)
		for frame in frame_iter:
			pending_frames.append(frame)
			if len(pending_frames) >= max(1, chunk_size):
				flush_pending(pending_frames)
				pending_frames = []
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		flush_pending(pending_frames)
		if written <= 0:
			raise ValueError("No frames provided")
	finally:
		if process is not None:
			_close_ffmpeg_encoder(process, audio_path)


def _prepare_source_face(
	source_cv2: np.ndarray,
	score_threshold: float,
	source_sort_order: str,
	source_face_index: int,
	face_detector_model: str
) -> Optional[Dict]:
	source_faces = detect_faces(
		source_cv2,
		score_threshold,
		source_sort_order,
		face_detector_model,
		with_embedding=True
	)
	if not source_faces:
		return None
	return source_faces[min(source_face_index, len(source_faces) - 1)]


def _prepare_reference_face(
	reference_image: Optional[Tensor],
	score_threshold: float,
	target_sort_order: str,
	face_detector_model: str
) -> Optional[Dict]:
	if reference_image is None:
		return None
	reference_cv2 = tensor_to_cv2(reference_image)
	reference_faces = detect_faces(
		reference_cv2,
		score_threshold,
		target_sort_order,
		face_detector_model,
		with_embedding=True
	)
	return reference_faces[0] if reference_faces else None


def _make_cv2_swap_fn(
	source_cv2: np.ndarray,
	face_swapper_model: FaceSwapperModel,
	pixel_boost: str,
	face_mask_blur: float,
	face_occluder_model: Optional[str],
	face_parser_model: Optional[str],
	face_selector_mode: str,
	source_face_index: int,
	target_face_index: int,
	source_sort_order: str,
	target_sort_order: str,
	score_threshold: float,
	face_detector_model: str,
	face_mask_types: Optional[list],
	face_mask_areas: Optional[list],
	face_mask_regions: Optional[list],
	face_mask_padding: tuple,
	reference_image: Optional[Tensor] = None,
	reference_face_distance: float = 0.6
):
	source_face = _prepare_source_face(
		source_cv2,
		score_threshold,
		source_sort_order,
		source_face_index,
		face_detector_model
	)
	if source_face is None:
		print("[FaceFusionVideo] No face detected in source image; returning original frames")

	reference_face = _prepare_reference_face(
		reference_image,
		score_threshold,
		target_sort_order,
		face_detector_model
	) if face_selector_mode == 'reference' else None

	if face_selector_mode == 'reference' and reference_face is None:
		print("[FaceFusionVideo] No reference face detected; returning original frames")

	prepared_source_embedding = None
	if source_face is not None:
		swapper = get_local_swapper(face_swapper_model)
		if swapper.model_session is None:
			swapper.initialize()
		model_type = swapper.model_config['type']
		prepared_source_embedding = swapper._prepare_source_embedding(source_face, source_face, model_type)

	def swap_frame(target_cv2: np.ndarray) -> np.ndarray:
		if source_face is None or (face_selector_mode == 'reference' and reference_face is None):
			return target_cv2
		result_cv2 = swap_faces_local(
			source_image=source_cv2,
			target_image=target_cv2,
			model_name=face_swapper_model,
			pixel_boost=pixel_boost,
			face_mask_blur=face_mask_blur,
			face_selector_mode=face_selector_mode,
			source_face_index=source_face_index,
			target_face_index=target_face_index,
			source_sort_order=source_sort_order,
			target_sort_order=target_sort_order,
			score_threshold=score_threshold,
			face_occluder_model=face_occluder_model,
			face_parser_model=face_parser_model,
			face_detector_model=face_detector_model,
			face_mask_types=face_mask_types,
			face_mask_areas=face_mask_areas,
			face_mask_regions=face_mask_regions,
			face_mask_padding=face_mask_padding,
			reference_face_distance=reference_face_distance,
			source_face=source_face,
			reference_face=reference_face,
			target_with_embedding=face_selector_mode == 'reference',
			target_use_cache=False,
			prepared_source_embedding=prepared_source_embedding
		)
		return result_cv2

	return swap_frame


class SwapFaceVideo:
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'source_images': (IO.IMAGE,),  # Changed to plural to support batches
				'face_swapper_model':
				(
					[
						'hyperswap_1a_256',
						'hyperswap_1b_256',
						'hyperswap_1c_256',
						'ghost_1_256',
						'ghost_2_256',
						'ghost_3_256',
						'hififace_unofficial_256',
						'inswapper_128',
						'inswapper_128_fp16',
						'blendswap_256',
						'simswap_256',
						'simswap_unofficial_512',
						'uniface_256'
					],
					{
						'default': 'hyperswap_1a_256'
					}
				),
				'face_detector_model':
				(
					['scrfd', 'retinaface', 'yolo_face', 'yunet', 'many'],
					{
						'default': 'scrfd'
					}
				),
				'max_workers':
				(
					'INT',
					{
						'default': 4,
						'min': 1,
						'max': 32
					}
				),
				'frame_rate':
				(
					'FLOAT',
					{
						'default': 24.0,
						'min': 1.0,
						'max': 240.0,
						'step': 0.1
					}
				),
				'chunk_size':
				(
					'INT',
					{
						'default': 60,
						'min': 1,
						'max': 10000
					}
				),
				'enable_nsfw_check':
				(
					'BOOLEAN',
					{
						'default': True
					}
				),
				'filename_prefix':
				(
					'STRING',
					{
						'default': 'FaceFusion/video',
						'multiline': False
					}
				)
			},
			'optional':
			{
				'target_frames': (IO.IMAGE,),
				'video_path':
				(
					'STRING',
					{
						'default': '',
						'multiline': False
					}
				),
				'target_audio': ('AUDIO',)
			}
		}

	RETURN_TYPES = ('STRING',)
	RETURN_NAMES = ('video_path',)
	FUNCTION = 'process'
	CATEGORY = 'FaceFusion'

	@staticmethod
	def process(source_images : Tensor, face_swapper_model : FaceSwapperModel, face_detector_model: str, max_workers : int, frame_rate: float, chunk_size: int, enable_nsfw_check: bool = True, filename_prefix: str = 'FaceFusion/video', target_frames: Optional[Tensor] = None, video_path: str = '', target_audio: Optional[Dict] = None) -> Tuple[str]:
		video_source = None
		try:
			output_path = _make_output_video_path(filename_prefix)
			# Handle multiple source images by taking the first one
			if source_images.dim() == 4 and source_images.shape[0] > 1:
				source_image = source_images[0:1]
			else:
				source_image = source_images

			video_source = _prepare_target_video_source(target_frames, video_path, frame_rate, target_audio)
			effective_frame_rate = video_source["frame_rate"]
			
			# Check source image for NSFW content (only if using local inference)
			source_cv2 = tensor_to_cv2(source_image)
			if enable_nsfw_check and CONTENT_FILTER_AVAILABLE:
				if analyse_frame(source_cv2):
					print("[ContentFilter] NSFW source detected in video - returning blurred video")
					_write_processed_video_output(
						video_source["frame_iter_factory"](),
						blur_frame,
						chunk_size,
						max_workers,
						effective_frame_rate,
						output_path,
						target_audio,
						video_source["audio_source_path"],
						video_source["width"],
						video_source["height"]
					)
					return (output_path,)
			
			# Sample check for NSFW in target video (check first, middle, last frame)
			if enable_nsfw_check and CONTENT_FILTER_AVAILABLE and len(video_source["sample_frames"]) > 0:
				nsfw_detected = False
				for frame_cv2 in video_source["sample_frames"]:
					if analyse_frame(frame_cv2):
						nsfw_detected = True
						break

				if nsfw_detected:
					print("[ContentFilter] NSFW content detected in target video - returning blurred video")
					_write_processed_video_output(
						video_source["frame_iter_factory"](),
						blur_frame,
						chunk_size,
						max_workers,
						effective_frame_rate,
						output_path,
						target_audio,
						video_source["audio_source_path"],
						video_source["width"],
						video_source["height"]
					)
					return (output_path,)

			swap_face = _make_cv2_swap_fn(
				source_cv2,
				face_swapper_model = face_swapper_model,
				pixel_boost = '512x512',
				face_mask_blur = 0.3,
				face_occluder_model = None,
				face_parser_model = None,
				face_selector_mode = 'one',
				source_face_index = 0,
				target_face_index = 0,
				source_sort_order = 'large-small',
				target_sort_order = 'large-small',
				score_threshold = 0.3,
				face_detector_model = face_detector_model,
				face_mask_types = ['box'],
				face_mask_areas = None,
				face_mask_regions = None,
				face_mask_padding = (0, 0, 0, 0)
			)

			_write_processed_video_output(
				video_source["frame_iter_factory"](),
				swap_face,
				chunk_size,
				max_workers,
				effective_frame_rate,
				output_path,
				target_audio,
				video_source["audio_source_path"],
				video_source["width"],
				video_source["height"]
			)
			return (output_path,)
				
		except RuntimeError as e:
			# Re-raise RuntimeError with clear message (don't return original video)
			print(f"[SwapFaceVideo] Fatal error: {e}")
			raise
		except Exception as e:
			print(f"[SwapFaceVideo] Unexpected error: {e}")
			import traceback
			traceback.print_exc()
			# Return original video on unexpected errors only
			output_path = _make_output_video_path(filename_prefix)
			if video_source is None:
				video_source = _prepare_target_video_source(target_frames, video_path, frame_rate, target_audio)
			_write_original_video_source(video_source, target_frames, output_path, frame_rate, target_audio)
			return (output_path,)




class AdvancedSwapFaceVideo:
	"""Advanced video face swapping node with face selection options."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'source_images': (IO.IMAGE,),
				'face_swapper_model':
				(
					[
						'hyperswap_1a_256',
						'hyperswap_1b_256',
						'hyperswap_1c_256',
						'ghost_1_256',
						'ghost_2_256',
						'ghost_3_256',
						'hififace_unofficial_256',
						'inswapper_128',
						'inswapper_128_fp16',
						'blendswap_256',
						'simswap_256',
						'simswap_unofficial_512',
						'uniface_256'
					],
					{
						'default': 'hyperswap_1a_256'
					}
				),
				'face_detector_model':
				(
					['scrfd', 'retinaface', 'yolo_face', 'yunet', 'many'],
					{
						'default': 'scrfd'
					}
				),
				'pixel_boost':
				(
					['256x256', '512x512', '768x768', '1024x1024'],
					{
						'default': '512x512'
					}
				),
			'face_occluder_model':
			(
				['none', 'xseg_1', 'xseg_2', 'xseg_3'],
				{
					'default': 'xseg_1'
				}
			),
			'face_parser_model':
			(
				['none', 'bisenet_resnet_18', 'bisenet_resnet_34'],
				{
					'default': 'bisenet_resnet_34'
				}
			),
				'face_mask_blur':
				(
					'FLOAT',
					{
						'default': 0.3,
						'min': 0.0,
						'max': 1.0,
						'step': 0.05
					}
				),
				'face_selector_mode':
				(
					['one', 'many', 'reference'],
					{
						'default': 'one'
					}
				),
				'source_face_index':
				(
					'INT',
					{
						'default': 0,
						'min': 0,
						'max': 100
					}
				),
				'target_face_index':
				(
					'INT',
					{
						'default': 0,
						'min': 0,
						'max': 100
					}
				),
				'source_sort_order':
				(
					['large-small', 'small-large', 'left-right', 'right-left', 'top-bottom', 'bottom-top', 'best-worst', 'worst-best'],
					{
						'default': 'large-small'
					}
				),
				'target_sort_order':
				(
					['large-small', 'small-large', 'left-right', 'right-left', 'top-bottom', 'bottom-top', 'best-worst', 'worst-best'],
					{
						'default': 'large-small'
					}
				),
				'score_threshold':
				(
					'FLOAT',
					{
					'default': 0.3,
						'min': 0.0,
						'max': 1.0,
						'step': 0.05
					}
				),
				'use_box_mask':
				(
					'BOOLEAN',
					{
						'default': True
					}
				),
				'use_occlusion_mask':
				(
					'BOOLEAN',
					{
						'default': False
					}
				),
				'use_area_mask':
				(
					'BOOLEAN',
					{
						'default': False
					}
				),
				'use_region_mask':
				(
					'BOOLEAN',
					{
						'default': False
					}
				),
				'face_mask_areas':
				(
					'STRING',
					{
						'default': 'upper-face,lower-face,mouth',
						'multiline': False
					}
				),
				'face_mask_regions':
				(
					'STRING',
					{
						'default': 'skin,nose,mouth,upper-lip,lower-lip',
						'multiline': False
					}
				),
				'face_mask_padding':
				(
					'STRING',
					{
						'default': '0,0,0,0',
						'multiline': False
					}
				),
				'max_workers':
				(
					'INT',
					{
						'default': 4,
						'min': 1,
						'max': 32
					}
				),
				'frame_rate':
				(
					'FLOAT',
					{
						'default': 24.0,
						'min': 1.0,
						'max': 240.0,
						'step': 0.1
					}
				),
				'chunk_size':
				(
					'INT',
					{
						'default': 60,
						'min': 1,
						'max': 10000
					}
				),
				'enable_nsfw_check':
				(
					'BOOLEAN',
					{
						'default': True
					}
				),
				'filename_prefix':
				(
					'STRING',
					{
						'default': 'FaceFusion/video',
						'multiline': False
					}
				)
			},
			'optional':
			{
				'target_frames': (IO.IMAGE,),
				'video_path':
				(
					'STRING',
					{
						'default': '',
						'multiline': False
					}
				),
				'target_audio': ('AUDIO',),
				'reference_image': (IO.IMAGE,),
				'reference_face_distance':
				(
					'FLOAT',
					{
						'default': 0.6,
						'min': 0.0,
						'max': 1.0,
						'step': 0.05
					}
				)
			}
		}
	
	RETURN_TYPES = ('STRING',)
	RETURN_NAMES = ('video_path',)
	FUNCTION = 'process'
	CATEGORY = 'FaceFusion'
	
	def process(
		self,
		source_images: Tensor,
		face_swapper_model: FaceSwapperModel,
		face_detector_model: str,
		pixel_boost: str,
		face_occluder_model: str,
		face_parser_model: str,
		face_mask_blur: float,
		face_selector_mode: str,
		source_face_index: int,
		target_face_index: int,
		source_sort_order: str,
		target_sort_order: str,
		score_threshold: float,
		use_box_mask: bool = True,
		use_occlusion_mask: bool = False,
		use_area_mask: bool = False,
		use_region_mask: bool = False,
		face_mask_areas: str = 'upper-face,lower-face,mouth',
		face_mask_regions: str = 'skin,nose,mouth,upper-lip,lower-lip',
		face_mask_padding: str = '0,0,0,0',
		max_workers: int = 4,
		frame_rate: float = 24.0,
		chunk_size: int = 60,
		enable_nsfw_check: bool = True,
		filename_prefix: str = 'FaceFusion/video',
		target_frames: Optional[Tensor] = None,
		video_path: str = '',
		target_audio: Optional[Dict] = None,
		reference_image: Optional[Tensor] = None,
		reference_face_distance: float = 0.6
	) -> Tuple[str]:
		"""Process video face swapping with advanced selection."""
		# Build face_mask_types list based on boolean options
		face_mask_types = []
		if use_box_mask:
			face_mask_types.append('box')
		if use_occlusion_mask:
			face_mask_types.append('occlusion')
		if use_area_mask:
			face_mask_types.append('area')
		if use_region_mask:
			face_mask_types.append('region')
		
		# Parse mask areas and regions from comma-separated strings
		mask_areas = [a.strip() for a in face_mask_areas.split(',') if a.strip()]
		mask_regions = [r.strip() for r in face_mask_regions.split(',') if r.strip()]
		
		# Parse padding (top, right, bottom, left)
		try:
			padding = tuple(int(p.strip()) for p in face_mask_padding.split(','))
			if len(padding) != 4:
				padding = (0, 0, 0, 0)
		except:
			padding = (0, 0, 0, 0)
		
		video_source = None
		try:
			output_path = _make_output_video_path(filename_prefix)
			# Handle multiple source images
			if source_images.dim() == 4 and source_images.shape[0] > 1:
				source_image = source_images[0:1]
			else:
				source_image = source_images

			video_source = _prepare_target_video_source(target_frames, video_path, frame_rate, target_audio)
			effective_frame_rate = video_source["frame_rate"]
			
			# Check source image for NSFW content (only if using local inference)
			source_cv2 = tensor_to_cv2(source_image)
			if enable_nsfw_check and CONTENT_FILTER_AVAILABLE:
				if analyse_frame(source_cv2):
					print("[ContentFilter] NSFW source detected in video - returning blurred video")
					_write_processed_video_output(
						video_source["frame_iter_factory"](),
						blur_frame,
						chunk_size,
						max_workers,
						effective_frame_rate,
						output_path,
						target_audio,
						video_source["audio_source_path"],
						video_source["width"],
						video_source["height"]
					)
					return (output_path,)
			
			# Sample check for NSFW in target video (check first, middle, last frame)
			if enable_nsfw_check and CONTENT_FILTER_AVAILABLE and len(video_source["sample_frames"]) > 0:
				nsfw_detected = False

				for frame_cv2 in video_source["sample_frames"]:
					if analyse_frame(frame_cv2):
						nsfw_detected = True
						break

				if nsfw_detected:
					print("[ContentFilter] NSFW content detected in target video - returning blurred video")
					_write_processed_video_output(
						video_source["frame_iter_factory"](),
						blur_frame,
						chunk_size,
						max_workers,
						effective_frame_rate,
						output_path,
						target_audio,
						video_source["audio_source_path"],
						video_source["width"],
						video_source["height"]
					)
					return (output_path,)

			swap_face = _make_cv2_swap_fn(
				source_cv2,
				face_swapper_model = face_swapper_model,
				pixel_boost = pixel_boost,
				face_mask_blur = face_mask_blur,
				face_occluder_model = face_occluder_model,
				face_parser_model = face_parser_model,
				face_selector_mode = face_selector_mode,
				source_face_index = source_face_index,
				target_face_index = target_face_index,
				source_sort_order = source_sort_order,
				target_sort_order = target_sort_order,
				score_threshold = score_threshold,
				face_detector_model = face_detector_model,
				face_mask_types = face_mask_types,
				face_mask_areas = mask_areas,
				face_mask_regions = mask_regions,
				face_mask_padding = padding,
				reference_image = reference_image,
				reference_face_distance = reference_face_distance
			)

			_write_processed_video_output(
				video_source["frame_iter_factory"](),
				swap_face,
				chunk_size,
				max_workers,
				effective_frame_rate,
				output_path,
				target_audio,
				video_source["audio_source_path"],
				video_source["width"],
				video_source["height"]
			)
			return (output_path,)
				
		except RuntimeError as e:
			# Re-raise RuntimeError with clear message (don't return original video)
			print(f"[AdvancedSwapFaceVideo] Fatal error: {e}")
			raise
		except Exception as e:
			print(f"[AdvancedSwapFaceVideo] Unexpected error: {e}")
			import traceback
			traceback.print_exc()
			# Return original video on unexpected errors only
			output_path = _make_output_video_path(filename_prefix)
			if video_source is None:
				video_source = _prepare_target_video_source(target_frames, video_path, frame_rate, target_audio)
			_write_original_video_source(video_source, target_frames, output_path, frame_rate, target_audio)
			return (output_path,)
