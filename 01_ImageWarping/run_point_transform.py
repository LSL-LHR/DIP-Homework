import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    ### FILL: Implement MLS or RBF based image warping

    if len(source_pts) < 2 or len(target_pts) < 2:
        return warped_image
    
    h, w = warped_image.shape[:2]
    n_points = min(len(source_pts), len(target_pts))
    p = source_pts[:n_points].astype(np.float64)
    q = target_pts[:n_points].astype(np.float64)
    
    # 创建网格
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).astype(np.float64)
    
    warped_pixels = np.zeros_like(pixels)
    batch_size = 5000
    n_pixels = pixels.shape[0]
    
    for start_idx in range(0, n_pixels, batch_size):
        end_idx = min(start_idx + batch_size, n_pixels)
        v = pixels[start_idx:end_idx]
        batch_len = v.shape[0]
        
        # 计算权重
        weights = np.zeros((batch_len, n_points))
        for i in range(n_points):
            diff = v - p[i]
            dist_sq = np.sum(diff ** 2, axis=1)
            weights[:, i] = 1.0 / (dist_sq ** alpha + eps)
        
        weight_sum = np.sum(weights, axis=1, keepdims=True) + eps
        weights = weights / weight_sum
        
        # 计算加权质心
        p_star = np.zeros((batch_len, 2))
        q_star = np.zeros((batch_len, 2))
        for i in range(n_points):
            p_star += weights[:, i:i+1] * p[i]
            q_star += weights[:, i:i+1] * q[i]
        
        # 中心化
        p_hat = p[np.newaxis, :, :] - p_star[:, np.newaxis, :]
        q_hat = q[np.newaxis, :, :] - q_star[:, np.newaxis, :]
        v_hat = v - p_star
        
        # 仿射变换
        for i in range(batch_len):
            P = p_hat[i].T
            Q = q_hat[i].T
            W = np.diag(weights[i])
            
            PW = P @ W
            PWP_T = PW @ P.T + eps * np.eye(2)
            
            try:
                M = Q @ W @ P.T @ np.linalg.inv(PWP_T)
                warped_pixels[start_idx + i] = M @ v_hat[i] + q_star[i]
            except:
                warped_pixels[start_idx + i] = v[i]
    
    # 重映射
    warped_x = warped_pixels[:, 0].reshape(h, w).astype(np.float32)
    warped_y = warped_pixels[:, 1].reshape(h, w).astype(np.float32)
    
    if len(warped_image.shape) == 3:
        for c in range(warped_image.shape[2]):
            warped_image[:, :, c] = cv2.remap(
                warped_image[:, :, c], warped_x, warped_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
    else:
        warped_image = cv2.remap(
            warped_image, warped_x, warped_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )


    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
