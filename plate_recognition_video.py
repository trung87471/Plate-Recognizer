import cv2

from plate_recognition_img import find_largest_rectangle, read_license_plate


def recognize_license_plates_on_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Không thể mở video tại đường dẫn: " + video_path)
        return

    while True:
        # Đọc từng frame của video
        ret, frame = cap.read()

        # Nếu không còn frame nào nữa thì thoát
        if not ret:
            break

        img = find_largest_rectangle(frame, False)

        if img is None:
            continue
        result = read_license_plate(img, False)
        if result == '':
            continue
        print('Biển số:')
        print(result)

        cv2.imshow('Frame', frame)
        cv2.putText(frame, result, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    video_path = r'asset/videos/1.mp4'
    recognize_license_plates_on_video(video_path)


if __name__ == '__main__':
    main()
