"""
LCARS Qwen Model Nodes
----------------------
Beautiful, futuristic nodes for Qwen language and vision models.

Supports:
- Qwen Text Generation
- Qwen Vision-Language (VL)
- Qwen Image Captioning
- Qwen Chat Interface
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from .lcars_base_node import LCARSModelNode, LCARSImageNode, register_node

# Optional imports with graceful fallback
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available - Qwen nodes will use fallback mode")


class QwenTextGenerator(LCARSModelNode):
    """
    🤖 Qwen Text Generation Node
    Generate text using Qwen language models.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Once upon a time",
                    "multiline": True
                }),
                "model_size": (["Qwen-7B", "Qwen-14B", "Qwen-72B", "Qwen2-7B"], {
                    "default": "Qwen2-7B"
                }),
                "max_tokens": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
            }
        }

    @classmethod
    def define_outputs(cls):
        return (("STRING",), ("text",))

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Qwen"

    def process(self, prompt, model_size, max_tokens, temperature, top_p, system_prompt=""):
        """Generate text using Qwen model"""

        if not TRANSFORMERS_AVAILABLE:
            return (f"[FALLBACK] Generated text for: {prompt[:50]}...",)

        try:
            # Load model
            model, tokenizer = self._load_qwen_model(model_size)

            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Tokenize
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Generate
            print(f"🤖 Generating text with {model_size}...")
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

            # Decode
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            print(f"✅ Generated {len(generated_text)} characters")
            return (generated_text,)

        except Exception as e:
            print(f"❌ Error in Qwen generation: {str(e)}")
            return (f"Error: {str(e)}",)

    def _load_qwen_model(self, model_size):
        """Load Qwen model with caching"""

        def load_fn():
            model_name = f"Qwen/{model_size}"
            print(f"📦 Loading {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            return model, tokenizer

        return self.load_model(model_size, load_fn)


class QwenVisionLanguage(LCARSModelNode):
    """
    👁️ Qwen Vision-Language Node
    Analyze images and answer questions using Qwen-VL.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {
                    "default": "What do you see in this image?",
                    "multiline": True
                }),
                "model_variant": (["Qwen-VL", "Qwen-VL-Chat"], {
                    "default": "Qwen-VL-Chat"
                }),
                "max_tokens": ("INT", {
                    "default": 200,
                    "min": 1,
                    "max": 1024,
                    "step": 1
                }),
            },
        }

    @classmethod
    def define_outputs(cls):
        return (("STRING", "IMAGE"), ("answer", "annotated_image"))

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Qwen"

    def process(self, image, question, model_variant, max_tokens):
        """Process image with Qwen-VL"""

        if not TRANSFORMERS_AVAILABLE:
            return (
                f"[FALLBACK] Analysis: This appears to be an image. Question: {question}",
                image
            )

        try:
            # Convert tensor to PIL
            pil_image = self.tensor_to_pil(image)

            # Load model
            model, tokenizer = self._load_qwen_vl_model(model_variant)

            # Prepare query
            query = tokenizer.from_list_format([
                {'image': pil_image},
                {'text': question},
            ])

            # Generate response
            print(f"👁️  Analyzing image with {model_variant}...")
            inputs = tokenizer(query, return_tensors='pt').to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens
            )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"✅ Generated answer: {answer[:100]}...")

            return (answer, image)

        except Exception as e:
            print(f"❌ Error in Qwen-VL: {str(e)}")
            return (f"Error: {str(e)}", image)

    def _load_qwen_vl_model(self, model_variant):
        """Load Qwen-VL model with caching"""

        def load_fn():
            model_name = f"Qwen/{model_variant}"
            print(f"📦 Loading {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            ).eval()

            return model, tokenizer

        return self.load_model(model_variant, load_fn)


class QwenImageCaption(LCARSModelNode):
    """
    📝 Qwen Image Captioning Node
    Generate detailed captions for images.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "caption_style": (
                    ["Detailed", "Brief", "Creative", "Technical"],
                    {"default": "Detailed"}
                ),
                "language": (["English", "Chinese", "Auto"], {
                    "default": "English"
                }),
            },
        }

    @classmethod
    def define_outputs(cls):
        return (("STRING",), ("caption",))

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Qwen"

    def process(self, image, caption_style, language):
        """Generate image caption"""

        # Style prompts
        style_prompts = {
            "Detailed": "Describe this image in detail, including objects, colors, composition, and mood.",
            "Brief": "Provide a brief, one-sentence description of this image.",
            "Creative": "Write a creative, poetic description of this image.",
            "Technical": "Provide a technical analysis of this image, including composition, lighting, and visual elements."
        }

        prompt = style_prompts.get(caption_style, style_prompts["Detailed"])

        if language == "Chinese":
            prompt = "用中文" + prompt

        if not TRANSFORMERS_AVAILABLE:
            return (f"[FALLBACK] Caption for image in {caption_style} style",)

        try:
            # Convert tensor to PIL
            pil_image = self.tensor_to_pil(image)

            # Load model
            model, tokenizer = self._load_qwen_vl_model("Qwen-VL-Chat")

            # Generate caption
            query = tokenizer.from_list_format([
                {'image': pil_image},
                {'text': prompt},
            ])

            print(f"📝 Generating {caption_style.lower()} caption...")
            inputs = tokenizer(query, return_tensors='pt').to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=300
            )

            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"✅ Caption: {caption[:100]}...")
            return (caption,)

        except Exception as e:
            print(f"❌ Error generating caption: {str(e)}")
            return (f"Error: {str(e)}",)

    def _load_qwen_vl_model(self, model_variant):
        """Load Qwen-VL model with caching"""

        def load_fn():
            model_name = f"Qwen/{model_variant}"
            print(f"📦 Loading {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            ).eval()

            return model, tokenizer

        return self.load_model(model_variant, load_fn)


class QwenChat(LCARSModelNode):
    """
    💬 Qwen Interactive Chat Node
    Multi-turn conversation with Qwen models.
    """

    # Class variable to store conversation history
    _conversations: Dict[str, List[Dict]] = {}

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "message": ("STRING", {
                    "default": "Hello! How can you help me?",
                    "multiline": True
                }),
                "conversation_id": ("STRING", {
                    "default": "default",
                }),
                "model_size": (["Qwen2-7B", "Qwen-14B"], {
                    "default": "Qwen2-7B"
                }),
                "clear_history": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "You are a helpful AI assistant.",
                    "multiline": True
                }),
            }
        }

    @classmethod
    def define_outputs(cls):
        return (("STRING", "STRING"), ("response", "history"))

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Qwen"

    def process(self, message, conversation_id, model_size, clear_history, system_prompt="You are a helpful AI assistant."):
        """Interactive chat with conversation history"""

        # Clear history if requested
        if clear_history:
            self._conversations[conversation_id] = []
            print(f"🗑️  Cleared conversation: {conversation_id}")

        # Initialize conversation if needed
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        # Add user message
        self._conversations[conversation_id].append({
            "role": "user",
            "content": message
        })

        if not TRANSFORMERS_AVAILABLE:
            response = f"[FALLBACK] Response to: {message}"
            self._conversations[conversation_id].append({
                "role": "assistant",
                "content": response
            })
            history = self._format_history(conversation_id)
            return (response, history)

        try:
            # Load model
            model, tokenizer = self._load_qwen_model(model_size)

            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self._conversations[conversation_id])

            # Generate
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            print(f"💬 Chatting with {model_size}...")
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )

            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Add assistant response to history
            self._conversations[conversation_id].append({
                "role": "assistant",
                "content": response
            })

            # Format history
            history = self._format_history(conversation_id)

            print(f"✅ Response: {response[:100]}...")
            return (response, history)

        except Exception as e:
            print(f"❌ Error in chat: {str(e)}")
            error_msg = f"Error: {str(e)}"
            return (error_msg, self._format_history(conversation_id))

    def _format_history(self, conversation_id):
        """Format conversation history as readable string"""
        if conversation_id not in self._conversations:
            return ""

        history_lines = []
        for msg in self._conversations[conversation_id]:
            role = msg["role"].upper()
            content = msg["content"]
            history_lines.append(f"[{role}]\n{content}\n")

        return "\n".join(history_lines)

    def _load_qwen_model(self, model_size):
        """Load Qwen model with caching"""

        def load_fn():
            model_name = f"Qwen/{model_size}"
            print(f"📦 Loading {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            return model, tokenizer

        return self.load_model(model_size, load_fn)


class QwenImageToText(LCARSImageNode):
    """
    🖼️→📝 Qwen Image-to-Text Overlay Node
    Generate text from image and optionally overlay it.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "What is in this image?",
                    "multiline": True
                }),
                "overlay_text": ("BOOLEAN", {
                    "default": False
                }),
                "text_color": (["White", "Black", "Orange", "Green"], {
                    "default": "Orange"
                }),
            },
        }

    @classmethod
    def define_outputs(cls):
        return (("IMAGE", "STRING"), ("image", "text"))

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Qwen"

    def process(self, image, prompt, overlay_text, text_color):
        """Generate text and optionally overlay on image"""

        # Generate text using Qwen-VL
        if TRANSFORMERS_AVAILABLE:
            try:
                pil_image = self.tensor_to_pil(image)
                model, tokenizer = self._load_qwen_vl()

                query = tokenizer.from_list_format([
                    {'image': pil_image},
                    {'text': prompt},
                ])

                inputs = tokenizer(query, return_tensors='pt')
                outputs = model.generate(**inputs, max_new_tokens=200)
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            except Exception as e:
                text = f"Error: {str(e)}"
        else:
            text = f"[FALLBACK] Generated text for image"

        # Overlay if requested
        if overlay_text:
            pil_image = self.tensor_to_pil(image)
            pil_image = self._overlay_text(pil_image, text, text_color)
            image = self.pil_to_tensor(pil_image)

        return (image, text)

    def _overlay_text(self, pil_image, text, color):
        """Overlay text on image with LCARS styling"""
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(pil_image)

        # Color mapping
        colors = {
            "White": (255, 255, 255),
            "Black": (0, 0, 0),
            "Orange": (255, 153, 0),
            "Green": (0, 255, 0),
        }
        text_color = colors.get(color, colors["Orange"])

        # Try to use a nice font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()

        # Word wrap
        words = text.split()
        lines = []
        current_line = []
        max_width = pil_image.width - 40

        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        # Draw background
        y = 20
        line_height = 30
        padding = 10

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Semi-transparent background
            draw.rectangle(
                [20 - padding, y - padding, 20 + text_width + padding, y + text_height + padding],
                fill=(0, 0, 0, 180)
            )

            # Text with glow effect
            for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((20 + offset[0], y + offset[1]), line, fill=(0, 0, 0), font=font)

            draw.text((20, y), line, fill=text_color, font=font)
            y += line_height

        return pil_image

    def _load_qwen_vl(self):
        """Load Qwen-VL model"""
        def load_fn():
            model_name = "Qwen/Qwen-VL-Chat"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()
            return model, tokenizer

        return self.load_model("Qwen-VL-Chat", load_fn)


# ==================== Node Registration ====================

NODE_CLASS_MAPPINGS = {
    "QwenTextGenerator": QwenTextGenerator,
    "QwenVisionLanguage": QwenVisionLanguage,
    "QwenImageCaption": QwenImageCaption,
    "QwenChat": QwenChat,
    "QwenImageToText": QwenImageToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTextGenerator": "🤖 Qwen Text Generator",
    "QwenVisionLanguage": "👁️ Qwen Vision-Language",
    "QwenImageCaption": "📝 Qwen Image Caption",
    "QwenChat": "💬 Qwen Chat",
    "QwenImageToText": "🖼️→📝 Qwen Image-to-Text",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
