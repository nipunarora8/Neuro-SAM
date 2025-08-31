import numpy as np
import random

class ContrastingColorManager:
    """
    Manages contrasting color pairs for dendrite-spine visualization.
    Each dendrite gets a muted color and its spines get a contrasting neon color.
    """
    
    def __init__(self):
        # Define 20 contrasting color pairs: (dendrite_color, spine_neon_color)
        # Each pair is designed to be visually contrasting and complementary
        self.color_pairs = [
            # Pair 1: Blue dendrite -> Orange neon spine
            ((0.2, 0.4, 0.8), (1.0, 0.5, 0.0)),
            
            # Pair 2: Green dendrite -> Magenta neon spine
            ((0.3, 0.7, 0.3), (1.0, 0.0, 1.0)),
            
            # Pair 3: Purple dendrite -> Yellow neon spine
            ((0.6, 0.3, 0.8), (1.0, 1.0, 0.0)),
            
            # Pair 4: Orange dendrite -> Cyan neon spine
            ((0.8, 0.5, 0.2), (0.0, 1.0, 1.0)),
            
            # Pair 5: Teal dendrite -> Red neon spine
            ((0.2, 0.7, 0.6), (1.0, 0.0, 0.0)),
            
            # Pair 6: Pink dendrite -> Green neon spine
            ((0.8, 0.4, 0.6), (0.0, 1.0, 0.0)),
            
            # Pair 7: Yellow-green dendrite -> Purple neon spine
            ((0.6, 0.8, 0.3), (0.5, 0.0, 1.0)),
            
            # Pair 8: Coral dendrite -> Blue neon spine
            ((0.8, 0.5, 0.4), (0.0, 0.5, 1.0)),
            
            # Pair 9: Indigo dendrite -> Orange-red neon spine
            ((0.3, 0.3, 0.7), (1.0, 0.3, 0.0)),
            
            # Pair 10: Olive dendrite -> Pink neon spine
            ((0.6, 0.6, 0.3), (1.0, 0.0, 0.5)),
            
            # Pair 11: Magenta dendrite -> Lime neon spine
            ((0.7, 0.3, 0.7), (0.5, 1.0, 0.0)),
            
            # Pair 12: Cyan dendrite -> Orange neon spine
            ((0.3, 0.7, 0.7), (1.0, 0.6, 0.0)),
            
            # Pair 13: Brown dendrite -> Cyan neon spine
            ((0.6, 0.4, 0.3), (0.0, 1.0, 0.8)),
            
            # Pair 14: Lavender dendrite -> Yellow-green neon spine
            ((0.7, 0.5, 0.8), (0.7, 1.0, 0.0)),
            
            # Pair 15: Navy dendrite -> Yellow neon spine
            ((0.2, 0.3, 0.6), (1.0, 0.9, 0.0)),
            
            # Pair 16: Mint dendrite -> Magenta neon spine
            ((0.4, 0.8, 0.6), (1.0, 0.2, 0.8)),
            
            # Pair 17: Maroon dendrite -> Cyan neon spine
            ((0.6, 0.2, 0.3), (0.2, 1.0, 1.0)),
            
            # Pair 18: Gold dendrite -> Blue neon spine
            ((0.8, 0.7, 0.3), (0.0, 0.3, 1.0)),
            
            # Pair 19: Slate dendrite -> Orange neon spine
            ((0.4, 0.5, 0.6), (1.0, 0.4, 0.0)),
            
            # Pair 20: Plum dendrite -> Green neon spine
            ((0.7, 0.4, 0.7), (0.2, 1.0, 0.2)),
        ]
        
        # Track which pairs have been used
        self.used_pair_indices = []
        
        # Track path assignments for consistent spine coloring
        self.path_pair_assignments = {}
    
    def get_contrasting_pair(self, path_id):
        """
        Get a contrasting color pair for a specific path.
        Returns the same pair if called multiple times for the same path.
        
        Args:
            path_id: Unique identifier for the path
            
        Returns:
            tuple: (dendrite_color, spine_neon_color)
        """
        # If this path already has an assignment, return it
        if path_id in self.path_pair_assignments:
            pair_index = self.path_pair_assignments[path_id]
            dendrite_color, spine_color = self.color_pairs[pair_index]
            print(f"Returning existing color pair for path {path_id}: dendrite={dendrite_color}, spine={spine_color}")
            return dendrite_color, spine_color
        
        # Find an unused color pair
        available_indices = [i for i in range(len(self.color_pairs)) if i not in self.used_pair_indices]
        
        if available_indices:
            chosen_index = random.choice(available_indices)
            self.used_pair_indices.append(chosen_index)
        else:
            # If all pairs used, reset and start over
            print("All color pairs used, resetting...")
            self.used_pair_indices = []
            chosen_index = random.randint(0, len(self.color_pairs) - 1)
            self.used_pair_indices.append(chosen_index)
        
        # Store the assignment for this path
        self.path_pair_assignments[path_id] = chosen_index
        
        dendrite_color, spine_color = self.color_pairs[chosen_index]
        
        print(f"Assigned new color pair {chosen_index} to path {path_id}:")
        print(f"  Dendrite color: {dendrite_color}")
        print(f"  Spine neon color: {spine_color}")
        
        return dendrite_color, spine_color
    
    def get_dendrite_color(self, path_id):
        """Get the dendrite color for a specific path"""
        dendrite_color, _ = self.get_contrasting_pair(path_id)
        return dendrite_color
    
    def get_spine_color(self, path_id):
        """Get the contrasting spine color for a specific path"""
        _, spine_color = self.get_contrasting_pair(path_id)
        return spine_color
    
    def reset_assignments(self):
        """Reset all color assignments (useful for new sessions)"""
        self.used_pair_indices = []
        self.path_pair_assignments = {}
        print("Color assignments reset")
    
    def get_pair_info(self, path_id):
        """Get information about the color pair assigned to a path"""
        if path_id in self.path_pair_assignments:
            pair_index = self.path_pair_assignments[path_id]
            dendrite_color, spine_color = self.color_pairs[pair_index]
            return {
                'pair_index': pair_index,
                'dendrite_color': dendrite_color,
                'spine_color': spine_color,
                'dendrite_hex': self._rgb_to_hex(dendrite_color),
                'spine_hex': self._rgb_to_hex(spine_color)
            }
        return None
    
    def _rgb_to_hex(self, rgb):
        """Convert RGB tuple to hex string for display"""
        r, g, b = [int(c * 255) for c in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def print_all_pairs(self):
        """Print all available color pairs for reference"""
        print("\nAvailable contrasting color pairs:")
        print("=" * 60)
        for i, (dendrite, spine) in enumerate(self.color_pairs):
            dendrite_hex = self._rgb_to_hex(dendrite)
            spine_hex = self._rgb_to_hex(spine)
            print(f"Pair {i+1:2d}: Dendrite {dendrite_hex} -> Spine {spine_hex}")
        print("=" * 60)

# Global instance
contrasting_color_manager = ContrastingColorManager()