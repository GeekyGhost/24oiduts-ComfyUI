"""
LCARS Procedural Generation Nodes
----------------------------------
Beautiful procedural generators for creating stunning visual effects.

Supports:
- Noise Generators (Perlin, Simplex, Fractal)
- Pattern Generators (Geometric, Organic, Sci-Fi)
- Gradient Generators
- Particle Systems
- Fractal Generators
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import math
from typing import Dict, List, Tuple, Any, Optional
from lcars_base_node import LCARSGeneratorNode, LCARSImageNode, register_node


class ProceduralNoiseGenerator(LCARSGeneratorNode):
    """
    🌫️ Procedural Noise Generator
    Generate various types of noise patterns.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "noise_type": (
                    ["Perlin", "Simplex", "White", "Pink", "Blue", "Fractal"],
                    {"default": "Perlin"}
                ),
                "scale": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
        }

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Procedural"

    def process(self, width, height, noise_type, scale, octaves, persistence, lacunarity, seed):
        """Generate noise pattern"""

        print(f"🌫️  Generating {noise_type} noise at {width}x{height}")

        np.random.seed(seed)

        if noise_type == "White":
            noise = self._white_noise(width, height)
        elif noise_type == "Pink":
            noise = self._pink_noise(width, height)
        elif noise_type == "Blue":
            noise = self._blue_noise(width, height)
        elif noise_type == "Perlin":
            noise = self._perlin_noise(width, height, scale, octaves, persistence, lacunarity)
        elif noise_type == "Simplex":
            noise = self._simplex_noise(width, height, scale, octaves, persistence, lacunarity)
        else:  # Fractal
            noise = self._fractal_noise(width, height, scale, octaves, persistence, lacunarity)

        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

        # Convert to RGB
        noise_rgb = np.stack([noise, noise, noise], axis=-1)

        # Convert to tensor
        tensor = torch.from_numpy(noise_rgb.astype(np.float32)).unsqueeze(0)

        print(f"✅ Generated noise pattern")
        return (tensor,)

    def _white_noise(self, width, height):
        """Generate white noise"""
        return np.random.rand(height, width)

    def _pink_noise(self, width, height):
        """Generate pink noise (1/f noise)"""
        # Generate white noise
        white = np.random.rand(height, width)

        # Apply FFT
        fft = np.fft.fft2(white)
        freq_x = np.fft.fftfreq(width)
        freq_y = np.fft.fftfreq(height)

        # Create frequency grid
        fx, fy = np.meshgrid(freq_x, freq_y)
        freq = np.sqrt(fx**2 + fy**2)
        freq[0, 0] = 1  # Avoid division by zero

        # Apply 1/f filter
        fft_filtered = fft / freq

        # Inverse FFT
        pink = np.real(np.fft.ifft2(fft_filtered))

        return pink

    def _blue_noise(self, width, height):
        """Generate blue noise approximation"""
        # Simple blue noise using high-pass filter
        white = np.random.rand(height, width)

        # Apply high-pass filter
        from scipy.ndimage import gaussian_filter
        lowpass = gaussian_filter(white, sigma=5)
        blue = white - lowpass

        return blue

    def _perlin_noise(self, width, height, scale, octaves, persistence, lacunarity):
        """Generate Perlin-like noise"""
        noise = np.zeros((height, width))
        amplitude = 1.0
        frequency = scale

        for _ in range(octaves):
            # Generate octave
            octave = self._generate_octave(width, height, frequency)
            noise += octave * amplitude

            amplitude *= persistence
            frequency *= lacunarity

        return noise

    def _simplex_noise(self, width, height, scale, octaves, persistence, lacunarity):
        """Generate Simplex-like noise (simplified)"""
        # Simplified version using Perlin with different interpolation
        return self._perlin_noise(width, height, scale, octaves, persistence, lacunarity)

    def _fractal_noise(self, width, height, scale, octaves, persistence, lacunarity):
        """Generate fractal noise"""
        noise = self._perlin_noise(width, height, scale, octaves, persistence, lacunarity)

        # Apply fractal transformation
        noise = np.abs(noise)
        noise = 1 - noise

        return noise

    def _generate_octave(self, width, height, frequency):
        """Generate a single noise octave"""
        # Create coordinate grid
        x = np.linspace(0, frequency * width, width)
        y = np.linspace(0, frequency * height, height)
        X, Y = np.meshgrid(x, y)

        # Generate smooth random values using sine waves
        noise = (
            np.sin(X) * np.cos(Y) +
            np.sin(X * 2.1 + 1.3) * np.cos(Y * 2.1 + 4.7) +
            np.sin(X * 4.3 + 2.1) * np.cos(Y * 4.3 + 1.9)
        )

        return noise


class ProceduralPatternGenerator(LCARSGeneratorNode):
    """
    🔷 Procedural Pattern Generator
    Generate geometric and organic patterns.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "pattern_type": (
                    ["Grid", "Hexagon", "Voronoi", "Circles", "Waves", "Spiral", "LCARS"],
                    {"default": "LCARS"}
                ),
                "scale": ("FLOAT", {"default": 50.0, "min": 5.0, "max": 200.0, "step": 5.0}),
                "complexity": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "color_scheme": (
                    ["Orange Glow", "Blue Neon", "Rainbow", "Monochrome"],
                    {"default": "Orange Glow"}
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
        }

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Procedural"

    def process(self, width, height, pattern_type, scale, complexity, color_scheme, seed):
        """Generate pattern"""

        print(f"🔷 Generating {pattern_type} pattern at {width}x{height}")

        np.random.seed(seed)

        # Create PIL image for drawing
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Get color palette
        colors = self._get_color_palette(color_scheme)

        # Generate pattern
        if pattern_type == "Grid":
            self._draw_grid(draw, width, height, scale, colors)
        elif pattern_type == "Hexagon":
            self._draw_hexagons(draw, width, height, scale, colors)
        elif pattern_type == "Voronoi":
            img = self._draw_voronoi(width, height, complexity, colors)
        elif pattern_type == "Circles":
            self._draw_circles(draw, width, height, scale, complexity, colors)
        elif pattern_type == "Waves":
            img = self._draw_waves(width, height, scale, complexity, colors)
        elif pattern_type == "Spiral":
            self._draw_spiral(draw, width, height, scale, complexity, colors)
        else:  # LCARS
            self._draw_lcars_pattern(draw, width, height, scale, colors)

        # Convert to tensor
        tensor = self.pil_to_tensor(img)

        print(f"✅ Generated {pattern_type} pattern")
        return (tensor,)

    def _get_color_palette(self, scheme):
        """Get color palette based on scheme"""
        palettes = {
            "Orange Glow": [(255, 153, 0), (204, 102, 0), (255, 204, 153)],
            "Blue Neon": [(0, 153, 255), (0, 102, 204), (153, 204, 255)],
            "Rainbow": [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (139, 0, 255)],
            "Monochrome": [(255, 255, 255), (200, 200, 200), (150, 150, 150)],
        }
        return palettes.get(scheme, palettes["Orange Glow"])

    def _draw_grid(self, draw, width, height, scale, colors):
        """Draw grid pattern"""
        spacing = int(scale)

        for x in range(0, width, spacing):
            draw.line([(x, 0), (x, height)], fill=colors[0], width=2)

        for y in range(0, height, spacing):
            draw.line([(0, y), (width, y)], fill=colors[0], width=2)

    def _draw_hexagons(self, draw, width, height, scale, colors):
        """Draw hexagon pattern"""
        size = int(scale / 2)
        h = size * math.sqrt(3)

        for row in range(int(height / h) + 2):
            for col in range(int(width / (1.5 * size)) + 2):
                x = col * 1.5 * size
                y = row * h + (h / 2 if col % 2 else 0)

                # Hexagon points
                points = []
                for i in range(6):
                    angle = math.pi / 3 * i
                    px = x + size * math.cos(angle)
                    py = y + size * math.sin(angle)
                    points.append((px, py))

                draw.polygon(points, outline=colors[col % len(colors)], width=2)

    def _draw_voronoi(self, width, height, num_points, colors):
        """Draw Voronoi diagram"""
        # Generate random points
        points = np.random.rand(num_points, 2)
        points[:, 0] *= width
        points[:, 1] *= height

        # Create distance field
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Find nearest point
                distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
                nearest_idx = np.argmin(distances)

                # Assign color
                color = colors[nearest_idx % len(colors)]
                img_array[y, x] = color

        return Image.fromarray(img_array)

    def _draw_circles(self, draw, width, height, scale, num_circles, colors):
        """Draw random circles"""
        for i in range(num_circles):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            r = np.random.randint(int(scale/2), int(scale*2))

            color = colors[i % len(colors)]
            draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=3)

    def _draw_waves(self, width, height, scale, complexity, colors):
        """Draw wave patterns"""
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                value = 0
                for i in range(complexity):
                    freq = (i + 1) * 0.01 / (scale / 50)
                    value += math.sin(x * freq + i) * math.cos(y * freq + i)

                # Normalize and map to color
                value = (value + complexity) / (2 * complexity)
                color_idx = int(value * (len(colors) - 1))
                img_array[y, x] = colors[color_idx]

        return Image.fromarray(img_array)

    def _draw_spiral(self, draw, width, height, scale, complexity, colors):
        """Draw spiral pattern"""
        cx, cy = width // 2, height // 2
        points = []

        for i in range(complexity * 100):
            t = i * 0.1
            r = scale * 0.1 * t
            x = cx + r * math.cos(t)
            y = cy + r * math.sin(t)

            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))

        # Draw spiral
        if len(points) > 1:
            for i in range(len(points) - 1):
                color = colors[i % len(colors)]
                draw.line([points[i], points[i+1]], fill=color, width=3)

    def _draw_lcars_pattern(self, draw, width, height, scale, colors):
        """Draw LCARS-style interface pattern"""
        # Top bar
        draw.rectangle([0, 0, width, int(scale)], fill=colors[0])

        # Side panels
        panel_width = int(scale * 2)
        draw.rounded_rectangle([0, int(scale*2), panel_width, height], radius=20, fill=colors[1])

        # Corner elements
        corner_size = int(scale * 1.5)
        draw.rounded_rectangle([width-corner_size, 0, width, corner_size], radius=15, fill=colors[0])
        draw.rounded_rectangle([0, height-corner_size, corner_size, height], radius=15, fill=colors[2])

        # Data bars
        for i in range(5):
            y = int(scale * 3 + i * scale * 1.2)
            bar_width = int(width * 0.3 * np.random.rand() + width * 0.2)
            draw.rounded_rectangle([panel_width+10, y, panel_width+bar_width, y+int(scale*0.8)],
                                  radius=10, fill=colors[i % len(colors)])


class ProceduralGradientGenerator(LCARSGeneratorNode):
    """
    🌈 Procedural Gradient Generator
    Generate beautiful gradient patterns.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "gradient_type": (
                    ["Linear", "Radial", "Angular", "Diamond", "Spiral", "Conic"],
                    {"default": "Radial"}
                ),
                "color1_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color1_g": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color1_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color2_r": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color2_g": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color2_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
            },
        }

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Procedural"

    def process(self, width, height, gradient_type, color1_r, color1_g, color1_b,
                color2_r, color2_g, color2_b, angle):
        """Generate gradient"""

        print(f"🌈 Generating {gradient_type} gradient at {width}x{height}")

        color1 = np.array([color1_r, color1_g, color1_b])
        color2 = np.array([color2_r, color2_g, color2_b])

        # Create coordinate grids
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)

        # Generate gradient based on type
        if gradient_type == "Linear":
            # Rotate coordinates
            angle_rad = math.radians(angle)
            gradient = X * math.cos(angle_rad) + Y * math.sin(angle_rad)

        elif gradient_type == "Radial":
            # Distance from center
            cx, cy = 0.5, 0.5
            gradient = np.sqrt((X - cx)**2 + (Y - cy)**2)

        elif gradient_type == "Angular":
            cx, cy = 0.5, 0.5
            gradient = np.arctan2(Y - cy, X - cx) / (2 * math.pi) + 0.5

        elif gradient_type == "Diamond":
            cx, cy = 0.5, 0.5
            gradient = np.abs(X - cx) + np.abs(Y - cy)

        elif gradient_type == "Spiral":
            cx, cy = 0.5, 0.5
            r = np.sqrt((X - cx)**2 + (Y - cy)**2)
            theta = np.arctan2(Y - cy, X - cx)
            gradient = (r + theta / (2 * math.pi)) % 1.0

        else:  # Conic
            cx, cy = 0.5, 0.5
            gradient = (np.arctan2(Y - cy, X - cx) + math.pi) / (2 * math.pi)

        # Normalize
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)

        # Apply colors
        gradient_rgb = np.zeros((height, width, 3))
        for c in range(3):
            gradient_rgb[:, :, c] = color1[c] * (1 - gradient) + color2[c] * gradient

        # Convert to tensor
        tensor = torch.from_numpy(gradient_rgb.astype(np.float32)).unsqueeze(0)

        print(f"✅ Generated gradient")
        return (tensor,)


class ProceduralFractalGenerator(LCARSGeneratorNode):
    """
    ❄️ Procedural Fractal Generator
    Generate stunning fractal patterns (Mandelbrot, Julia, etc.).
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "fractal_type": (
                    ["Mandelbrot", "Julia", "Burning Ship", "Newton"],
                    {"default": "Mandelbrot"}
                ),
                "max_iterations": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "center_x": ("FLOAT", {"default": -0.5, "min": -2.0, "max": 2.0, "step": 0.1}),
                "center_y": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "julia_c_real": ("FLOAT", {"default": -0.7, "min": -2.0, "max": 2.0, "step": 0.01}),
                "julia_c_imag": ("FLOAT", {"default": 0.27, "min": -2.0, "max": 2.0, "step": 0.01}),
            },
        }

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Procedural"

    def process(self, width, height, fractal_type, max_iterations, zoom,
                center_x, center_y, julia_c_real, julia_c_imag):
        """Generate fractal"""

        print(f"❄️  Generating {fractal_type} fractal at {width}x{height}")

        # Create complex plane
        x_min = center_x - 2.0 / zoom
        x_max = center_x + 2.0 / zoom
        y_min = center_y - 2.0 / zoom
        y_max = center_y + 2.0 / zoom

        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        if fractal_type == "Mandelbrot":
            fractal = self._mandelbrot(C, max_iterations)
        elif fractal_type == "Julia":
            julia_c = julia_c_real + 1j * julia_c_imag
            fractal = self._julia(C, julia_c, max_iterations)
        elif fractal_type == "Burning Ship":
            fractal = self._burning_ship(C, max_iterations)
        else:  # Newton
            fractal = self._newton(C, max_iterations)

        # Colorize
        fractal_rgb = self._colorize_fractal(fractal, max_iterations)

        # Convert to tensor
        tensor = torch.from_numpy(fractal_rgb.astype(np.float32)).unsqueeze(0)

        print(f"✅ Generated fractal")
        return (tensor,)

    def _mandelbrot(self, C, max_iter):
        """Compute Mandelbrot set"""
        Z = np.zeros_like(C)
        M = np.zeros(C.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] = i

        return M

    def _julia(self, Z, c, max_iter):
        """Compute Julia set"""
        M = np.zeros(Z.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + c
            M[mask] = i

        return M

    def _burning_ship(self, C, max_iter):
        """Compute Burning Ship fractal"""
        Z = np.zeros_like(C)
        M = np.zeros(C.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag))**2 + C[mask]
            M[mask] = i

        return M

    def _newton(self, Z, max_iter):
        """Compute Newton fractal for z^3 - 1 = 0"""
        M = np.zeros(Z.shape)

        for i in range(max_iter):
            mask = np.abs(Z**3 - 1) > 1e-6
            Z[mask] = Z[mask] - (Z[mask]**3 - 1) / (3 * Z[mask]**2)
            M[mask] = i

        return M

    def _colorize_fractal(self, fractal, max_iter):
        """Colorize fractal with LCARS color scheme"""
        # Normalize
        fractal = fractal / max_iter

        # Create RGB
        rgb = np.zeros((*fractal.shape, 3))

        # Orange glow color scheme
        rgb[:, :, 0] = fractal  # Red channel
        rgb[:, :, 1] = fractal * 0.6  # Green channel
        rgb[:, :, 2] = fractal * 0.0  # Blue channel

        # Add some variation
        rgb[:, :, 0] = np.sin(fractal * math.pi * 2) * 0.5 + 0.5
        rgb[:, :, 1] = np.sin(fractal * math.pi * 2 + math.pi/3) * 0.3 + 0.4
        rgb[:, :, 2] = np.sin(fractal * math.pi * 2 + 2*math.pi/3) * 0.2 + 0.1

        return rgb


# ==================== Node Registration ====================

NODE_CLASS_MAPPINGS = {
    "ProceduralNoiseGenerator": ProceduralNoiseGenerator,
    "ProceduralPatternGenerator": ProceduralPatternGenerator,
    "ProceduralGradientGenerator": ProceduralGradientGenerator,
    "ProceduralFractalGenerator": ProceduralFractalGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProceduralNoiseGenerator": "🌫️ Procedural Noise",
    "ProceduralPatternGenerator": "🔷 Procedural Pattern",
    "ProceduralGradientGenerator": "🌈 Procedural Gradient",
    "ProceduralFractalGenerator": "❄️ Procedural Fractal",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
