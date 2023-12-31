import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'tesseract\tesseract.exe'

video_mode = True  # Nếu video_mode = True thì sẽ không hiển thị ảnh


def show_image(title, image):
    if video_mode:
        return
    cv2.imshow(title, image)
    cv2.waitKey(0)


def preprocess_image(image):
    show_image('Anh goc', image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image('Anh xam', gray_image)
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    show_image('Anh nhi phan', binary_image)
    contours, hierarchy = cv2.findContours(binary_image, 1, 1)  # Tìm đường viền

    tmp_image = image.copy()  # Tạo bản sao của ảnh gốc
    cv2.drawContours(tmp_image, contours, -1, (0, 255, 0), 1)  # Vẽ đường viền lên ảnh gốc
    show_image('Ve duong vien len anh goc', tmp_image)

    largest_rectangle = [0, 0]
    for contour in contours:
        length = 0.01 * cv2.arcLength(contour, True)  # Xác định độ dài của đường viền
        approx = cv2.approxPolyDP(contour, length, True)  # Xác định đa giác xấp xỉ với đường viền
        if len(approx) == 4:  # Nếu đa giác có 4 cạnh thì đó là hình chữ nhật
            area = cv2.contourArea(contour)  # Tính diện tích của hình chữ nhật
            if area > largest_rectangle[0]:  # Tình diện tích lớn nhất
                largest_rectangle = [cv2.contourArea(contour), contour, approx]  # Lưu lại hình chữ nhật lớn nhất

    if largest_rectangle[0] == 0 and largest_rectangle[1] == 0:
        return None

    col, row, width, height = cv2.boundingRect(largest_rectangle[1])  # Xác định tọa độ của hình chữ nhật

    cv2.drawContours(image, [largest_rectangle[1]], 0, (0, 255, 0), 1)
    show_image('Dinh vi bien so xe tren anh', image)
    cropped_img = image[row:row + height, col:col + width]  # Cắt ảnh bằng tọa độ của hình chữ nhật
    show_image('Bien so xe', cropped_img)
    cropped_img = gray_image[row:row + height, col:col + width]  # Cắt ảnh xám bằng tọa độ của hình chữ nhật
    return cropped_img


def read_license_plate(img, enable_img):
    global video_mode
    video_mode = not enable_img
    img = preprocess_image(img)
    if img is None:
        return ''
    blur = cv2.GaussianBlur(img, (3, 3), 0)  # lam mo anh de giam nhieu va lam gon bien so
    binary_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (3, 3))  # tao kernel de mo rong anh nhi phan de nhan dang bien so
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)  # mo rong
    data = pytesseract.image_to_string(opening, lang='eng',
                                       config='--psm 6')  # su dung pytesseract de nhan dang bien so

    show_image('Bien so la', opening)
    return data


def main():
    global video_mode
    video_mode = False
    image_path = r'asset/images/1.png'
    image = cv2.imread(image_path)
    assert image is not None, 'Không thể đọc ảnh tại đường dẫn: ' + image_path
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    print('Bien so la:')
    res = read_license_plate(image, True)
    print(res)
    cv2.putText(image, res, (20, 40), cv2.QT_FONT_NORMAL, 1, (255, 255, 255))
    cv2.imshow('Ket qua', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
