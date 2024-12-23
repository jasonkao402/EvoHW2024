import numpy as np
import cv2

def generate_ring_mask(nodes, image_size=720, thickness=30, blur_kernel_size=15, blur_repeats=1):
    """
    Generates a ring-like alpha mask for a given path on a 720x720 image.
    
    Parameters:
        nodes (list of tuples): 2D coordinates of the path nodes (e.g., [(x1, y1), (x2, y2), ...]).
        image_size (int): Size of the square image (default is 720).
        thickness (int): Thickness of the ring.
        blur_kernel_size (int): Size of the Gaussian blur kernel for smoothing the edges.
        
    Returns:
        np.ndarray: Alpha mask as a 2D NumPy array.
    """
    # Create an empty black canvas
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Convert nodes to integer coordinates
    int_nodes = [(int(x), int(y)) for x, y in nodes]
    
    # Draw the closed path
    cv2.polylines(mask, [np.array(int_nodes, dtype=np.int32)], isClosed=True, color=255, thickness=thickness)

    # Apply Gaussian blur to soften edges
    blurred_mask = mask.copy()
    for _ in range(blur_repeats):
        blurred_mask = cv2.GaussianBlur(blurred_mask, (blur_kernel_size, blur_kernel_size), 0)
    
    return blurred_mask

# Example usage
nodes = [(200, 200), (500, 200), (500, 500), (200, 500), (100, 350)]
ring_mask = generate_ring_mask(nodes, image_size=700, blur_repeats=10)
print(np.min(ring_mask), np.max(ring_mask))  # 0 255
# Save or visualize the mask
cv2.imwrite("ring_mask.png", ring_mask)
cv2.imshow("Ring Mask", ring_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
