import cv2
import numpy as np

def sort_vertices(vertices, h, w):
    image_corners = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype=np.float32)
    corner_candidates = []
    for i in range(4):
        corner_candidates.append(np.roll(vertices, i, axis=0))

    displacements = []
    for corners in corner_candidates:
        corners = np.array(corners, dtype=np.float32)
        displacement = 0
        for corner, image_corner in zip(corners, image_corners):
            displacement += np.linalg.norm(corner - image_corner)
        displacements.append(displacement)

    min_displacement_index = np.argmin(displacements)

    best_corners = corner_candidates[min_displacement_index]

    return np.array(best_corners, dtype=np.float32)

def get_board_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_areas = [cv2.contourArea(contour) for contour in contours]
    largest_contour_indices = np.argsort(contour_areas)[-3:]
    largest_contours = [contours[i] for i in largest_contour_indices]

    largest_contours.sort(key=cv2.contourArea, reverse=True)

    for contour in largest_contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        approx = approx[:, 0, :]
        if len(approx) == 4:
            return approx

    return np.array([[0, 0], [0, img.shape[0]-1], [img.shape[1]-1, img.shape[0]-1], [img.shape[1]-1, 0]], dtype=np.float32)

def warp_image(img, origin_points, h, w):
    dest_points = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
    perspective_matrix = cv2.getPerspectiveTransform(origin_points, dest_points)
    transformed = cv2.warpPerspective(img, perspective_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return transformed

def warp_to_board(img):
    corners = get_board_corners(img)
    h, w, _ = img.shape
    origin_points = sort_vertices(corners, h, w)
    img = warp_image(img, origin_points, h, w)
    return img

def get_masks(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    image_s = image[:, :, 1]
    _, orange = cv2.threshold(image_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image_v = image[:, :, 2]
    thresh, _ = cv2.threshold(image_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = int(thresh * 0.85)

    gray = np.where(image_v >= thresh, 255, 0).astype(np.uint8)
    gray = cv2.bitwise_not(gray) 

    border = np.zeros_like(gray)
    border[10:-10, 10:-10] = 255

    gray = cv2.bitwise_and(gray, gray, mask=border)
    gray = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(orange))
    gray = cv2.dilate(gray, (3, 3), iterations=3)
    gray = cv2.erode(gray, (3, 3), iterations=3)
    
    orange = cv2.bitwise_and(orange, orange, mask=border)

    return orange, gray
    
    
def mask_square(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask

    min_x, min_y = mask.shape[1], mask.shape[0]
    max_x, max_y = 0, 0

    # Initialize variables to find the center of all contours
    total_x, total_y = 0, 0
    total_contours = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

        # Calculate the center of the contour
        center_x = x + w // 2
        center_y = y + h // 2

        # Accumulate center coordinates
        total_x += center_x
        total_y += center_y
        total_contours += 1

    # Calculate the center of all contours
    if total_contours > 0:
        center_x = total_x // total_contours
        center_y = total_y // total_contours
    else:
        # No contours found, use the center of the image
        center_x = mask.shape[1] // 2
        center_y = mask.shape[0] // 2

    # Calculate the maximum distance from the center to the contour's edge
    max_distance = max(max_x - center_x, center_x - min_x, max_y - center_y, center_y - min_y)

    # Calculate the size of the square based on the maximum distance
    size = 2 * max_distance

    # Ensure the square region is fully contained within the image
    if center_x - size // 2 < 0:
        center_x = size // 2
    if center_y - size // 2 < 0:
        center_y = size // 2
    if center_x + size // 2 > mask.shape[1]:
        center_x = mask.shape[1] - size // 2
    if center_y + size // 2 > mask.shape[0]:
        center_y = mask.shape[0] - size // 2

    # Calculate the coordinates for the square region
    x_square = center_x - size // 2
    y_square = center_y - size // 2

    return mask[y_square:y_square+size, x_square:x_square+size]


def preprocess(image, img_size=(28, 28), r=False):
    if isinstance(image, str):
        image = cv2.imread(image)

    image = warp_to_board(image)
    image = cv2.resize(image, (224, 224))

    orange, gray = get_masks(image)
    if r: image = gray
    else: image = orange

    image = mask_square(image)
    image = cv2.resize(image, img_size)
    image = cv2.erode(image, (3, 3), iterations=1)
    image = cv2.dilate(image, (3, 3), iterations=1)
    _, image = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)
    return image

def get_perspective_matrix(img):
    corners = get_board_corners(img)
    h, w, _ = img.shape
    origin_points = sort_vertices(corners, h, w)
    dest_points = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
    perspective_matrix = cv2.getPerspectiveTransform(origin_points, dest_points)
    return perspective_matrix

async def visualize_transform(img, output_path):
    try:
        h, w, _ = img.shape
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

        animation_duration_sec = 1
        frame_rate = 24

        total_frames = int(animation_duration_sec * frame_rate)

        initial_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        final_matrix = get_perspective_matrix(img)

        for frame_number in range(total_frames):
            interpolation_factor = frame_number / (total_frames - 1)

            matrix = (1 - interpolation_factor) * initial_matrix + interpolation_factor * final_matrix

            output_image = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

            out.write(output_image)

        img = warp_to_board(img)

        main, gray = get_masks(img)
        

        border = np.zeros_like(main)
        margin = int(w * 0.05)
        border[margin:-margin, margin:-margin] = 255

        main = cv2.bitwise_and(main, main, mask=border)
        main = cv2.dilate(main, (5, 5), iterations=3)

        gray = cv2.bitwise_and(gray, gray, mask=border)
        gray = cv2.dilate(gray, (5, 5), iterations=3)

        target_image = np.zeros_like(img)
        target_image[main == 255] = (0, 165, 255)
        target_image[gray == 255] = (255, 0, 0)

        # transition to target_image 
        for frame_number in range(total_frames):
            interpolation_factor = frame_number / (total_frames - 1)

            interpolated_image = cv2.addWeighted(img, 1 - interpolation_factor, target_image, interpolation_factor, 0)

            out.write(interpolated_image)
        
        # wait 1 sec on the target image
        for frame_number in range(frame_rate):
            out.write(target_image)

        out.release()
    except Exception as e:
        raise e