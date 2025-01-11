import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
POINTS = [0, 11, 12, 23, 24]

image_path = r"images\false\20250108_112440_frame_0000.jpg"
image = cv2.imread(image_path)

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
        
        selected_points = {index: (landmarks[index].x, landmarks[index].y) for index in POINTS}
   
        mid_shoulder = (
            (selected_points[11][0] + selected_points[12][0]) / 2,
            (selected_points[11][1] + selected_points[12][1]) / 2
        )
        mid_hip = (
            (selected_points[23][0] + selected_points[24][0]) / 2,
            (selected_points[23][1] + selected_points[24][1]) / 2
        )

        center = (
            (selected_points[11][0] + selected_points[12][0] + selected_points[23][0] + selected_points[24][0]) / 4,
            (selected_points[11][1] + selected_points[12][1] + selected_points[23][1] + selected_points[24][1]) / 4
        )

        for index, (x, y) in selected_points.items():
            cx, cy = int(x * w), int(y * h)  
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            # cv2.putText(image, str(index), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for point, label, color in zip(
            [mid_shoulder, mid_hip, center],
            ["Mid Shoulder", "Mid Hip", "Center"],
            [(255, 0, 0), (0, 0, 255), (0, 255, 255)] 
        ):
            px, py = int(point[0] * w), int(point[1] * h)
            cv2.circle(image, (px, py), 5, color, -1)
            # cv2.putText(image, label, (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Pose with Midpoints and Center", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
