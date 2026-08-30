def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [center_x, center_y, width, height] anchor boxes.
    """
    # Write code here
    import numpy as np
    
    res = []
    stride = image_size/feature_size
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j+0.5)*stride
            cy = (i+0.5)*stride

            for s in scales:
                for r in aspect_ratios:

                    w = s*np.sqrt(r)
                    h = s/np.sqrt(r)
        
                    res.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return res
            