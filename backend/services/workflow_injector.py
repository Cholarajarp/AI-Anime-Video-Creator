"""
Template → AnimateDiff Workflow Injection
Injects static PNG templates into AnimateDiff motion
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class WorkflowInjector:
    """Inject template latents into ComfyUI AnimateDiff workflows"""

    def __init__(self):
        self.node_id_counter = 1000

    def inject_template_into_workflow(
        self,
        workflow_json: Dict[str, Any],
        template_latents: Dict[str, Any],
        frame_count: int,
        context_length: int = 16,
        overlap: int = 4
    ) -> Dict[str, Any]:
        """Modify AnimateDiff workflow to use template latents"""

        workflow = copy.deepcopy(workflow_json)

        # Find and replace EmptyLatentImage node
        empty_latent_node_id = self._find_node_by_class(workflow, "EmptyLatentImage")
        if empty_latent_node_id:
            workflow = self._replace_with_template_latent(
                workflow,
                empty_latent_node_id,
                template_latents
            )

        # Insert ControlNet nodes for background stability
        if "background" in template_latents:
            workflow = self._add_background_controlnet(
                workflow,
                template_latents["background"]
            )

        # Configure AnimateDiff context
        animatediff_node_id = self._find_node_by_class(
            workflow,
            "AnimateDiffLoaderWithContext"
        )
        if animatediff_node_id:
            workflow = self._configure_context_window(
                workflow,
                animatediff_node_id,
                frame_count,
                context_length,
                overlap
            )

        # Ensure seed stability
        workflow = self._stabilize_seeds(workflow)

        return workflow

    def _find_node_by_class(
        self,
        workflow: Dict[str, Any],
        class_type: str
    ) -> Optional[str]:
        """Find node ID by class type"""
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict):
                if node_data.get("class_type") == class_type:
                    return node_id
        return None

    def _find_all_nodes_by_class(
        self,
        workflow: Dict[str, Any],
        class_type: str
    ) -> List[str]:
        """Find all node IDs by class type"""
        nodes = []
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict):
                if node_data.get("class_type") == class_type:
                    nodes.append(node_id)
        return nodes

    def _replace_with_template_latent(
        self,
        workflow: Dict[str, Any],
        node_id: str,
        template_latents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Replace EmptyLatentImage with TemplateLatentLoader"""

        original_node = workflow[node_id]

        # Create new template latent node
        new_node_id = str(self.node_id_counter)
        self.node_id_counter += 1

        workflow[new_node_id] = {
            "class_type": "TemplateLatentLoader",
            "inputs": {
                "character_latent": template_latents.get("character"),
                "background_latent": template_latents.get("background"),
                "width": original_node.get("inputs", {}).get("width", 512),
                "height": original_node.get("inputs", {}).get("height", 512),
                "batch_size": original_node.get("inputs", {}).get("batch_size", 16)
            }
        }

        # Update all connections pointing to old node
        for other_id, other_node in workflow.items():
            if isinstance(other_node, dict) and "inputs" in other_node:
                for input_name, input_value in other_node["inputs"].items():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        if input_value[0] == node_id:
                            other_node["inputs"][input_name] = [new_node_id, input_value[1]]

        del workflow[node_id]

        return workflow

    def _add_background_controlnet(
        self,
        workflow: Dict[str, Any],
        background_latent: Any
    ) -> Dict[str, Any]:
        """Add ControlNet node for static background"""

        controlnet_node_id = str(self.node_id_counter)
        self.node_id_counter += 1

        workflow[controlnet_node_id] = {
            "class_type": "ControlNetApply",
            "inputs": {
                "image": background_latent,
                "control_net": "control_v11f1e_sd15_tile",
                "strength": 0.8,
                "start_percent": 0.0,
                "end_percent": 1.0
            }
        }

        return workflow

    def _configure_context_window(
        self,
        workflow: Dict[str, Any],
        node_id: str,
        frame_count: int,
        context_length: int,
        overlap: int
    ) -> Dict[str, Any]:
        """Configure AnimateDiff context window for temporal coherence"""

        if node_id in workflow:
            workflow[node_id]["inputs"].update({
                "context_length": context_length,
                "context_overlap": overlap,
                "context_schedule": "uniform",
                "closed_loop": False,
                "video_length": frame_count
            })

        return workflow

    def _stabilize_seeds(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure consistent seeds for reproducibility"""

        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict) and "inputs" in node_data:
                if "seed" in node_data["inputs"]:
                    # Use fixed seed or make it controllable
                    if node_data["inputs"]["seed"] == -1:
                        node_data["inputs"]["seed"] = 42
                    node_data["inputs"]["control_after_generate"] = "fixed"

        return workflow

    def create_base_animatediff_workflow(
        self,
        prompt: str,
        negative_prompt: str = "bad quality, blurry, distorted",
        width: int = 512,
        height: int = 512,
        frame_count: int = 32,
        steps: int = 20,
        cfg: float = 7.5,
        seed: int = 42,
        checkpoint: str = "sd15_anime.safetensors",
        motion_module: str = "mm_sd15_v3.safetensors"
    ) -> Dict[str, Any]:
        """Create base AnimateDiff workflow structure"""

        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": checkpoint
                }
            },
            "2": {
                "class_type": "AnimateDiffLoaderWithContext",
                "inputs": {
                    "model": ["1", 0],
                    "motion_module": motion_module,
                    "context_length": 16,
                    "context_overlap": 4,
                    "context_schedule": "uniform"
                }
            },
            "3": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": frame_count
                }
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1]
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["4", 0],
                    "negative": ["6", 0],
                    "latent_image": ["3", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1]
                }
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "8": {
                "class_type": "VideoCombine",
                "inputs": {
                    "images": ["7", 0],
                    "frame_rate": 15,
                    "loop_count": 0,
                    "filename_prefix": "AnimateDiff",
                    "format": "video/h264-mp4"
                }
            }
        }

    def update_workflow_prompt(
        self,
        workflow: Dict[str, Any],
        prompt: str,
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update prompts in an existing workflow"""

        workflow = copy.deepcopy(workflow)

        # Find positive prompt nodes
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict):
                if node_data.get("class_type") == "CLIPTextEncode":
                    inputs = node_data.get("inputs", {})
                    # Check if this is positive or negative
                    text = inputs.get("text", "")
                    if "bad" in text.lower() or "blur" in text.lower() or "negative" in text.lower():
                        if negative_prompt:
                            inputs["text"] = negative_prompt
                    else:
                        inputs["text"] = prompt

        return workflow

    def update_workflow_settings(
        self,
        workflow: Dict[str, Any],
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        frame_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update various settings in workflow"""

        workflow = copy.deepcopy(workflow)

        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue

            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})

            if class_type == "KSampler":
                if seed is not None:
                    inputs["seed"] = seed
                if steps is not None:
                    inputs["steps"] = steps
                if cfg is not None:
                    inputs["cfg"] = cfg

            elif class_type == "EmptyLatentImage":
                if width is not None:
                    inputs["width"] = width
                if height is not None:
                    inputs["height"] = height
                if frame_count is not None:
                    inputs["batch_size"] = frame_count

            elif class_type == "AnimateDiffLoaderWithContext":
                if frame_count is not None:
                    inputs["video_length"] = frame_count

        return workflow

    def load_workflow_from_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load workflow JSON from file"""
        try:
            path = Path(filepath)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load workflow: {e}")
        return None

    def save_workflow_to_file(
        self,
        workflow: Dict[str, Any],
        filepath: str
    ) -> bool:
        """Save workflow JSON to file"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(workflow, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save workflow: {e}")
            return False


def inject_template_into_workflow(
    workflow_json: Dict[str, Any],
    template_latents: Dict[str, Any],
    frame_count: int
) -> Dict[str, Any]:
    """Convenience function for workflow injection"""
    injector = WorkflowInjector()
    return injector.inject_template_into_workflow(
        workflow_json, template_latents, frame_count
    )

