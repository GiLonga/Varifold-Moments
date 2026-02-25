import cv2
import numpy as np

def extract_contour_from_mask(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found. Check the path.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert because mask is black and background is white
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        raise ValueError("No contour found.")

    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour.reshape(-1, 2)

def ensure_ccw(contours):
    """
    Takes a list of contours (each as Nx2 array) and ensures they are counterclockwise.
    
    Args:
        contours (list of np.ndarray): List of contours
    
    Returns:
        list of np.ndarray: List of CCW contours
    """
    ccw_contours = []
    
    for contour in contours:
        # Compute signed area
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])  # shoelace formula
        # For closed polygon, also include last -> first
        area += 0.5 * (x[-1]*y[0] - x[0]*y[-1])
        
        if area < 0:  # CW → reverse to make CCW
            contour = contour[::-1]
        
        ccw_contours.append(contour)
    
    return ccw_contours