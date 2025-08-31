import random
import colorsys

def generate_random_color(exclude_neon=True, saturation_range=(0.4, 0.8), brightness_range=(0.5, 0.9)):
    """
    Generate a random color for dendrite segmentation masks.
    
    Args:
        exclude_neon: If True, avoid very bright/saturated colors (reserved for spines)
        saturation_range: Range for color saturation (0.0 to 1.0)
        brightness_range: Range for color brightness (0.0 to 1.0)
    
    Returns:
        Tuple of (R, G, B) values in range 0.0-1.0
    """
    # Generate random hue (0-360 degrees)
    hue = random.uniform(0, 1)
    
    # Generate saturation and brightness within specified ranges
    if exclude_neon:
        # More muted colors for dendrites
        saturation = random.uniform(saturation_range[0], saturation_range[1])
        brightness = random.uniform(brightness_range[0], brightness_range[1])
    else:
        # Allow full range including neon colors
        saturation = random.uniform(0.0, 1.0)
        brightness = random.uniform(0.0, 1.0)
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
    
    return rgb

def get_neon_colors():
    """
    Get a list of predefined neon colors for spine segmentation.
    
    Returns:
        List of (R, G, B) tuples for neon colors
    """
    neon_colors = [
        (0.0, 1.0, 0.0),    # Neon green
        (1.0, 0.0, 1.0),    # Neon magenta
        (0.0, 1.0, 1.0),    # Neon cyan
        (1.0, 1.0, 0.0),    # Neon yellow
        (1.0, 0.27, 0.0),   # Neon orange-red
        (0.5, 0.0, 1.0),    # Neon purple
        (0.0, 0.5, 1.0),    # Neon blue
        (1.0, 0.0, 0.5),    # Neon pink
        (0.5, 1.0, 0.0),    # Neon lime
        (1.0, 0.5, 0.0),    # Neon orange
    ]
    return neon_colors

def get_next_neon_color(index=None):
    """
    Get the next neon color from the predefined list.
    
    Args:
        index: Optional index to get specific color, if None uses random
        
    Returns:
        (R, G, B) tuple for neon color
    """
    neon_colors = get_neon_colors()
    
    if index is None:
        return random.choice(neon_colors)
    else:
        return neon_colors[index % len(neon_colors)]

class ColorManager:
    """
    Manages colors for different types of segmentation masks.
    """
    
    def __init__(self):
        self.used_dendrite_colors = []
        self.neon_color_index = 0
        
    def get_dendrite_color(self):
        """
        Get a unique random color for dendrite segmentation.
        Ensures no duplicate colors are used.
        
        Returns:
            (R, G, B) tuple
        """
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            color = generate_random_color(exclude_neon=True)
            
            # Check if this color is too similar to existing ones
            is_unique = True
            for used_color in self.used_dendrite_colors:
                # Calculate color distance (simple Euclidean distance in RGB space)
                distance = sum((a - b) ** 2 for a, b in zip(color, used_color)) ** 0.5
                if distance < 0.3:  # Minimum distance threshold
                    is_unique = False
                    break
            
            if is_unique:
                self.used_dendrite_colors.append(color)
                return color
                
            attempts += 1
        
        # If we can't find a unique color, just return a random one
        fallback_color = generate_random_color(exclude_neon=True)
        self.used_dendrite_colors.append(fallback_color)
        return fallback_color
    
    def get_spine_color(self):
        """
        Get the next neon color for spine segmentation.
        
        Returns:
            (R, G, B) tuple
        """
        color = get_next_neon_color(self.neon_color_index)
        self.neon_color_index += 1
        return color
    
    def reset_dendrite_colors(self):
        """Reset the used dendrite colors list."""
        self.used_dendrite_colors = []
    
    def reset_spine_colors(self):
        """Reset the spine color index."""
        self.neon_color_index = 0

# Global color manager instance
color_manager = ColorManager()