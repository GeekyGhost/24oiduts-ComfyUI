"""
LCARS Node Template Generator
------------------------------
Super easy system for users to create custom nodes!

Just fill in a simple template and get a fully functional node.
"""

import os
from typing import Dict, Any


class NodeTemplateGenerator:
    """
    🎯 Easy Node Template Generator

    Usage:
        1. Create a config dict with your node specifications
        2. Call generate() to create the node file
        3. Your node is ready to use!
    """

    TEMPLATE = '''"""
{description}
Generated with LCARS Node Template Generator
"""

from lcars_base_node import {base_class}
import torch
import numpy as np


class {class_name}({base_class}):
    """
    {emoji} {display_name}
    {description}
    """

    @classmethod
    def define_inputs(cls):
        return {inputs}

    @classmethod
    def define_outputs(cls):
        return {outputs}

    @classmethod
    def category(cls):
        return "{category}"

    def process(self, {process_params}):
        """
        {process_description}
        """
        print(f"{emoji} Processing {display_name}...")

        # YOUR CODE HERE
        {process_code}

        print(f"✅ Completed {display_name}")
        return {return_statement}


# ==================== Node Registration ====================

NODE_CLASS_MAPPINGS = {{
    "{class_name}": {class_name},
}}

NODE_DISPLAY_NAME_MAPPINGS = {{
    "{class_name}": "{emoji} {display_name}",
}}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
'''

    @classmethod
    def generate(cls, config: Dict[str, Any], output_path: str = None):
        """
        Generate a node from configuration.

        Args:
            config: Node configuration dict (see examples below)
            output_path: Where to save the file (optional)

        Returns:
            Generated code as string
        """

        # Extract config with defaults
        class_name = config.get("class_name", "CustomNode")
        display_name = config.get("display_name", class_name)
        emoji = config.get("emoji", "⭐")
        description = config.get("description", "Custom node")
        category = config.get("category", "Studio42/LCARS/Custom")
        base_class = config.get("base_class", "LCARSImageNode")

        # Inputs
        inputs = config.get("inputs", {
            "required": {
                "image": ("IMAGE",),
            }
        })

        # Outputs
        outputs = config.get("outputs", (("IMAGE",), ("output",)))

        # Process function
        process_params = cls._extract_process_params(inputs)
        process_code = config.get("process_code", "result = image\n        ")
        process_description = config.get("process_description", "Process the inputs")
        return_statement = config.get("return_statement", "(result,)")

        # Generate code
        code = cls.TEMPLATE.format(
            class_name=class_name,
            display_name=display_name,
            emoji=emoji,
            description=description,
            category=category,
            base_class=base_class,
            inputs=cls._format_dict(inputs, indent=2),
            outputs=cls._format_tuple(outputs),
            process_params=process_params,
            process_code=process_code,
            process_description=process_description,
            return_statement=return_statement,
        )

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(code)
            print(f"✅ Generated node at: {output_path}")

        return code

    @staticmethod
    def _extract_process_params(inputs: Dict) -> str:
        """Extract parameter names from inputs dict"""
        params = []

        for param_name in inputs.get("required", {}).keys():
            params.append(param_name)

        for param_name in inputs.get("optional", {}).keys():
            params.append(f"{param_name}=None")

        return ", ".join(params)

    @staticmethod
    def _format_dict(d: Dict, indent: int = 0) -> str:
        """Format dict for code generation"""
        import json
        json_str = json.dumps(d, indent=4)

        # Add indentation
        if indent > 0:
            lines = json_str.split('\n')
            indented = '\n'.join(' ' * (indent * 4) + line for line in lines)
            return indented

        return json_str

    @staticmethod
    def _format_tuple(t: tuple) -> str:
        """Format tuple for code generation"""
        return str(t)


# ==================== Example Configs ====================

EXAMPLE_BRIGHTNESS_NODE = {
    "class_name": "MyBrightnessNode",
    "display_name": "My Brightness Adjuster",
    "emoji": "☀️",
    "description": "Adjusts image brightness",
    "category": "Studio42/LCARS/MyNodes",
    "base_class": "LCARSImageNode",
    "inputs": {
        "required": {
            "image": ("IMAGE",),
            "brightness": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.1
            }),
        }
    },
    "outputs": (("IMAGE",), ("image",)),
    "process_code": """# Adjust brightness
        result = image * brightness
        result = torch.clamp(result, 0.0, 1.0)
        """,
    "return_statement": "(result,)",
}

EXAMPLE_TEXT_NODE = {
    "class_name": "MyTextProcessor",
    "display_name": "My Text Processor",
    "emoji": "📝",
    "description": "Process text in cool ways",
    "category": "Studio42/LCARS/MyNodes",
    "base_class": "LCARSBaseNode",
    "inputs": {
        "required": {
            "text": ("STRING", {"default": "Hello World"}),
            "operation": (["Uppercase", "Lowercase", "Reverse"], {"default": "Uppercase"}),
        }
    },
    "outputs": (("STRING",), ("processed_text",)),
    "process_code": """# Process text
        if operation == "Uppercase":
            result = text.upper()
        elif operation == "Lowercase":
            result = text.lower()
        else:
            result = text[::-1]
        """,
    "return_statement": "(result,)",
}

EXAMPLE_GENERATOR_NODE = {
    "class_name": "MyColorGenerator",
    "display_name": "My Color Generator",
    "emoji": "🎨",
    "description": "Generate solid color images",
    "category": "Studio42/LCARS/MyNodes",
    "base_class": "LCARSGeneratorNode",
    "inputs": {
        "required": {
            "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            "red": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "green": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            "blue": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
    },
    "outputs": (("IMAGE",), ("image",)),
    "process_code": """# Create solid color image
        result = self.create_blank_tensor(width, height, channels=3, fill_value=0.0)
        result[:, :, :, 0] = red
        result[:, :, :, 1] = green
        result[:, :, :, 2] = blue
        """,
    "return_statement": "(result,)",
}


# ==================== Quick Generate Functions ====================

def quick_image_node(name: str,
                     emoji: str,
                     description: str,
                     extra_inputs: Dict = None,
                     process_code: str = "result = image",
                     output_path: str = None):
    """
    Quickly generate an image processing node.

    Args:
        name: Node name (CamelCase)
        emoji: Emoji for the node
        description: What does this node do?
        extra_inputs: Additional inputs beyond 'image'
        process_code: Your processing code
        output_path: Where to save

    Returns:
        Generated code
    """

    inputs = {
        "required": {
            "image": ("IMAGE",),
        }
    }

    if extra_inputs:
        inputs["required"].update(extra_inputs)

    config = {
        "class_name": name,
        "display_name": name,
        "emoji": emoji,
        "description": description,
        "base_class": "LCARSImageNode",
        "category": "Studio42/LCARS/Custom",
        "inputs": inputs,
        "outputs": (("IMAGE",), ("image",)),
        "process_code": process_code,
        "return_statement": "(result,)",
    }

    return NodeTemplateGenerator.generate(config, output_path)


def quick_text_node(name: str,
                    emoji: str,
                    description: str,
                    inputs: Dict = None,
                    process_code: str = "result = 'Hello'",
                    output_path: str = None):
    """
    Quickly generate a text processing node.

    Args:
        name: Node name
        emoji: Emoji
        description: Description
        inputs: Input specifications
        process_code: Processing code
        output_path: Where to save

    Returns:
        Generated code
    """

    if inputs is None:
        inputs = {
            "required": {
                "text": ("STRING", {"default": ""}),
            }
        }

    config = {
        "class_name": name,
        "display_name": name,
        "emoji": emoji,
        "description": description,
        "base_class": "LCARSBaseNode",
        "category": "Studio42/LCARS/Custom",
        "inputs": inputs,
        "outputs": (("STRING",), ("text",)),
        "process_code": process_code,
        "return_statement": "(result,)",
    }

    return NodeTemplateGenerator.generate(config, output_path)


def quick_generator_node(name: str,
                        emoji: str,
                        description: str,
                        inputs: Dict = None,
                        process_code: str = "result = self.create_blank_tensor(512, 512)",
                        output_path: str = None):
    """
    Quickly generate a generator node (creates images from scratch).

    Args:
        name: Node name
        emoji: Emoji
        description: Description
        inputs: Input specifications
        process_code: Generation code
        output_path: Where to save

    Returns:
        Generated code
    """

    if inputs is None:
        inputs = {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }

    config = {
        "class_name": name,
        "display_name": name,
        "emoji": emoji,
        "description": description,
        "base_class": "LCARSGeneratorNode",
        "category": "Studio42/LCARS/Generators",
        "inputs": inputs,
        "outputs": (("IMAGE",), ("image",)),
        "process_code": process_code,
        "return_statement": "(result,)",
    }

    return NodeTemplateGenerator.generate(config, output_path)


# ==================== Interactive Generator ====================

def interactive_generate():
    """Interactive node generator - asks user questions"""

    print("🎯 LCARS Node Template Generator")
    print("=" * 50)

    # Ask questions
    class_name = input("Node class name (CamelCase): ").strip() or "MyCustomNode"
    display_name = input("Display name: ").strip() or class_name
    emoji = input("Emoji (optional): ").strip() or "⭐"
    description = input("Description: ").strip() or "Custom node"

    print("\nNode type:")
    print("1. Image processor (takes image, outputs image)")
    print("2. Generator (creates image from scratch)")
    print("3. Text processor")
    print("4. Custom")

    node_type = input("Choose (1-4): ").strip() or "1"

    if node_type == "1":
        code = quick_image_node(
            name=class_name,
            emoji=emoji,
            description=description,
            extra_inputs={
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            },
            process_code="result = image * strength\n        result = torch.clamp(result, 0.0, 1.0)",
        )

    elif node_type == "2":
        code = quick_generator_node(
            name=class_name,
            emoji=emoji,
            description=description,
        )

    elif node_type == "3":
        code = quick_text_node(
            name=class_name,
            emoji=emoji,
            description=description,
        )

    else:
        # Custom - use brightness example as template
        config = EXAMPLE_BRIGHTNESS_NODE.copy()
        config["class_name"] = class_name
        config["display_name"] = display_name
        config["emoji"] = emoji
        config["description"] = description

        code = NodeTemplateGenerator.generate(config)

    # Show and save
    print("\n" + "=" * 50)
    print("Generated Code:")
    print("=" * 50)
    print(code)
    print("=" * 50)

    save = input("\nSave to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("Filename: ").strip() or f"my_{class_name.lower()}.py"
        with open(filename, 'w') as f:
            f.write(code)
        print(f"✅ Saved to {filename}")


# ==================== Main ====================

if __name__ == "__main__":
    # Example: Generate the brightness node
    print("Generating example brightness node...")

    code = NodeTemplateGenerator.generate(
        EXAMPLE_BRIGHTNESS_NODE,
        output_path="example_brightness_node.py"
    )

    print("\nGenerated brightness node!")
    print("\nTo create your own node, import this module and use:")
    print("  - quick_image_node()")
    print("  - quick_text_node()")
    print("  - quick_generator_node()")
    print("  - NodeTemplateGenerator.generate()")
    print("\nOr run interactive_generate() for a guided experience!")
