"""
ComfyUI Client - WebSocket-based API Controller
================================================
This module provides a robust client for interacting with ComfyUI's API.
It handles workflow submission, progress tracking, and result retrieval.
"""

import json
import uuid
import asyncio
import websockets
import httpx
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class GenerationResult:
    """Result from a ComfyUI generation task."""
    success: bool
    output_files: list[str]
    prompt_id: str
    execution_time: float
    error_message: Optional[str] = None


class ComfyUIClient:
    """
    Asynchronous client for ComfyUI API communication.

    This client implements the WebSocket protocol for real-time progress
    tracking and uses HTTP for workflow submission and file retrieval.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        timeout: int = 600
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws?clientId={self.client_id}"

    async def check_connection(self) -> bool:
        """Verify ComfyUI server is accessible."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/system_stats",
                    timeout=10
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"ComfyUI connection check failed: {e}")
            return False

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get ComfyUI system statistics including VRAM usage."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/system_stats")
            return response.json()

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/queue")
            return response.json()

    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get generation history for a specific prompt."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/history/{prompt_id}")
            return response.json()

    async def upload_image(self, image_path: Path, subfolder: str = "") -> str:
        """Upload an image to ComfyUI for use in workflows."""
        async with httpx.AsyncClient() as client:
            with open(image_path, 'rb') as f:
                files = {'image': (image_path.name, f, 'image/png')}
                data = {'subfolder': subfolder, 'type': 'input'}
                response = await client.post(
                    f"{self.base_url}/upload/image",
                    files=files,
                    data=data
                )
                result = response.json()
                return result.get('name', '')

    async def queue_prompt(
        self,
        workflow: Dict[str, Any],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> GenerationResult:
        """
        Submit a workflow to ComfyUI and wait for completion.

        Args:
            workflow: The workflow JSON in API format
            progress_callback: Optional callback for progress updates (node_name, progress)

        Returns:
            GenerationResult with output files and execution info
        """
        import time
        start_time = time.time()

        # Submit the prompt
        async with httpx.AsyncClient() as http_client:
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }
            response = await http_client.post(
                f"{self.base_url}/prompt",
                json=payload
            )

            if response.status_code != 200:
                return GenerationResult(
                    success=False,
                    output_files=[],
                    prompt_id="",
                    execution_time=0,
                    error_message=f"Failed to queue prompt: {response.text}"
                )

            result = response.json()
            prompt_id = result.get('prompt_id', '')

        # Connect to WebSocket for progress tracking
        output_files = []
        error_message = None

        try:
            async with websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10
            ) as ws:
                async for message in asyncio.wait_for(
                    self._listen_for_completion(ws, prompt_id, progress_callback),
                    timeout=self.timeout
                ):
                    if message['type'] == 'execution_success':
                        # Fetch output files from history
                        history = await self.get_history(prompt_id)
                        if prompt_id in history:
                            outputs = history[prompt_id].get('outputs', {})
                            for node_id, node_output in outputs.items():
                                if 'gifs' in node_output:
                                    for gif in node_output['gifs']:
                                        output_files.append(gif['filename'])
                                if 'images' in node_output:
                                    for img in node_output['images']:
                                        output_files.append(img['filename'])
                        break
                    elif message['type'] == 'execution_error':
                        error_message = message.get('data', {}).get('exception_message', 'Unknown error')
                        break

        except asyncio.TimeoutError:
            error_message = f"Generation timed out after {self.timeout} seconds"
        except Exception as e:
            error_message = str(e)
            logger.error(f"WebSocket error: {e}")

        execution_time = time.time() - start_time

        return GenerationResult(
            success=error_message is None and len(output_files) > 0,
            output_files=output_files,
            prompt_id=prompt_id,
            execution_time=execution_time,
            error_message=error_message
        )

    async def _listen_for_completion(
        self,
        ws,
        prompt_id: str,
        progress_callback: Optional[Callable]
    ):
        """Generator that yields WebSocket messages until completion."""
        async for message in ws:
            data = json.loads(message)
            msg_type = data.get('type', '')

            if msg_type == 'executing':
                exec_data = data.get('data', {})
                if exec_data.get('prompt_id') == prompt_id:
                    node = exec_data.get('node')
                    if node is None:
                        # Execution complete
                        yield {'type': 'execution_success', 'data': data}
                    elif progress_callback:
                        progress_callback(f"Executing: {node}", 0.5)

            elif msg_type == 'progress':
                prog_data = data.get('data', {})
                if progress_callback:
                    value = prog_data.get('value', 0)
                    max_val = prog_data.get('max', 1)
                    progress = value / max_val if max_val > 0 else 0
                    progress_callback("Generating...", progress)

            elif msg_type == 'execution_error':
                exec_data = data.get('data', {})
                if exec_data.get('prompt_id') == prompt_id:
                    yield {'type': 'execution_error', 'data': exec_data}

    async def download_output(
        self,
        filename: str,
        output_dir: Path,
        subfolder: str = ""
    ) -> Path:
        """Download a generated file from ComfyUI."""
        async with httpx.AsyncClient() as client:
            params = {
                'filename': filename,
                'subfolder': subfolder,
                'type': 'output'
            }
            response = await client.get(
                f"{self.base_url}/view",
                params=params
            )

            output_path = output_dir / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                f.write(response.content)

            return output_path

    async def cancel_prompt(self, prompt_id: str) -> bool:
        """Cancel a running or queued generation."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/interrupt"
            )
            return response.status_code == 200

    async def clear_queue(self) -> bool:
        """Clear all pending generations from queue."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/queue",
                json={"clear": True}
            )
            return response.status_code == 200


class WorkflowBuilder:
    """
    Builder for constructing ComfyUI workflows programmatically.

    This class provides a fluent interface for building AnimateDiff
    video generation workflows.
    """

    def __init__(self):
        self.workflow = {}
        self._node_counter = 0

    def _next_id(self) -> str:
        self._node_counter += 1
        return str(self._node_counter)

    def add_checkpoint_loader(
        self,
        ckpt_name: str
    ) -> str:
        """Add a checkpoint loader node."""
        node_id = self._next_id()
        self.workflow[node_id] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": ckpt_name
            }
        }
        return node_id

    def add_clip_text_encode(
        self,
        text: str,
        clip_source: tuple[str, int]
    ) -> str:
        """Add a CLIP text encoding node."""
        node_id = self._next_id()
        self.workflow[node_id] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": text,
                "clip": clip_source
            }
        }
        return node_id

    def add_empty_latent(
        self,
        width: int,
        height: int,
        batch_size: int
    ) -> str:
        """Add an empty latent image node (for frame count)."""
        node_id = self._next_id()
        self.workflow[node_id] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": batch_size
            }
        }
        return node_id

    def add_animatediff_loader(
        self,
        model_source: tuple[str, int],
        motion_module: str = "mm_sd_v15_v2.ckpt",
        context_length: int = 16,
        context_overlap: int = 4
    ) -> str:
        """Add AnimateDiff motion module loader."""
        node_id = self._next_id()
        self.workflow[node_id] = {
            "class_type": "ADE_AnimateDiffLoaderWithContext",
            "inputs": {
                "model": model_source,
                "model_name": motion_module,
                "beta_schedule": "sqrt_linear (AnimateDiff)",
                "context_options": None
            }
        }
        return node_id

    def add_ksampler(
        self,
        model_source: tuple[str, int],
        positive_source: tuple[str, int],
        negative_source: tuple[str, int],
        latent_source: tuple[str, int],
        seed: int = -1,
        steps: int = 20,
        cfg: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        denoise: float = 1.0
    ) -> str:
        """Add KSampler node for diffusion."""
        import random
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        node_id = self._next_id()
        self.workflow[node_id] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_source,
                "positive": positive_source,
                "negative": negative_source,
                "latent_image": latent_source,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise
            }
        }
        return node_id

    def add_vae_decode(
        self,
        samples_source: tuple[str, int],
        vae_source: tuple[str, int]
    ) -> str:
        """Add VAE decode node."""
        node_id = self._next_id()
        self.workflow[node_id] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": samples_source,
                "vae": vae_source
            }
        }
        return node_id

    def add_video_combine(
        self,
        images_source: tuple[str, int],
        frame_rate: int = 15,
        format: str = "video/h264-mp4"
    ) -> str:
        """Add video combine node for final output."""
        node_id = self._next_id()
        self.workflow[node_id] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": images_source,
                "frame_rate": frame_rate,
                "loop_count": 0,
                "filename_prefix": "AnimateDiff",
                "format": format,
                "pingpong": False,
                "save_output": True
            }
        }
        return node_id

    def build(self) -> Dict[str, Any]:
        """Return the constructed workflow."""
        return self.workflow.copy()

    @classmethod
    def create_animatediff_workflow(
        cls,
        prompt: str,
        negative_prompt: str,
        checkpoint: str,
        motion_module: str,
        frame_count: int,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        fps: int = 15,
        seed: int = -1
    ) -> Dict[str, Any]:
        """
        Create a complete AnimateDiff workflow.

        This is a convenience method that builds a standard video
        generation workflow with all necessary nodes.
        """
        builder = cls()

        # Load checkpoint
        ckpt_id = builder.add_checkpoint_loader(checkpoint)

        # Encode prompts
        pos_id = builder.add_clip_text_encode(prompt, (ckpt_id, 1))
        neg_id = builder.add_clip_text_encode(negative_prompt, (ckpt_id, 1))

        # Create latent
        latent_id = builder.add_empty_latent(width, height, frame_count)

        # Load AnimateDiff
        motion_id = builder.add_animatediff_loader(
            (ckpt_id, 0),
            motion_module
        )

        # Sample
        sampler_id = builder.add_ksampler(
            (motion_id, 0),
            (pos_id, 0),
            (neg_id, 0),
            (latent_id, 0),
            seed=seed,
            steps=steps,
            cfg=cfg
        )

        # Decode
        decode_id = builder.add_vae_decode(
            (sampler_id, 0),
            (ckpt_id, 2)
        )

        # Combine to video
        builder.add_video_combine(
            (decode_id, 0),
            frame_rate=fps
        )

        return builder.build()

