import cv2
import torch
import mediapipe as mp

try:
    # YOLOv5sモデル読み込み（事前に git clone yolov5 済み）
    model = torch.hub.load('C:/Users/kurau/Bench_pose/yolov5', 'yolov5s', source='local')  # ローカルのパスを指定
  # source=localで高速化

    # MediaPipe pose 初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # 動画読み込み
    video_path = 'input.mp4'
    cap = cv2.VideoCapture(video_path)

    # 動画保存設定（任意：出力ファイルに可視化結果を保存）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_pose.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOで人物検出
        results = model(frame)

        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # personクラス
                x1, y1, x2, y2 = map(int, box)
                roi = frame[y1:y2, x1:x2]

                # MediaPipeに渡す
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_roi)

                if result.pose_landmarks:
                    # 座標変換（ROI → 元フレーム）
                    for lm in result.pose_landmarks.landmark:
                        cx = int(lm.x * (x2 - x1)) + x1
                        cy = int(lm.y * (y2 - y1)) + y1
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                    # 可視化（任意）
                    mp_drawing.draw_landmarks(
                        roi,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

        # 表示（確認用）
        cv2.imshow('Pose Detection', frame)
        out.write(frame)  # 保存
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error occurred: {e}")
