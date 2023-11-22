import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'tesseract\tesseract.exe'

video_mode = True


def show_img(title, img):
    if video_mode:
        return
    cv2.imshow(title, img)
    cv2.waitKey(0)


def find_largest_rectangle(img, enable_img):
    global video_mode
    video_mode = not enable_img
    show_img('Ảnh gốc', img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_img('Ảnh xám', gray_img)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    show_img('Ảnh nhị phân', binary_img)
    contours, hierarchy = cv2.findContours(binary_img, 1, 1)

    tmp_img = img.copy()
    cv2.drawContours(tmp_img, contours, -1, (0, 255, 0), 1)
    show_img('Vẽ đường viền lên ảnh gốc', tmp_img)

    largest_rectangle = [0, 0]
    for contour in contours:
        length = 0.01 * cv2.arcLength(contour, True)  # Xác định độ dài của đường viền
        approx = cv2.approxPolyDP(contour, length, True)  # Xác định đa giác xấp xỉ với đường viền
        if len(approx) == 4:  # Nếu đa giác có 4 cạnh thì đó là hình chữ nhật
            area = cv2.contourArea(contour)  # Tính diện tích của hình chữ nhật
            if area > largest_rectangle[0]:  # Tình diện tích lớn nhất
                largest_rectangle = [cv2.contourArea(contour), contour, approx]  # Lưu lại hình chữ nhật lớn nhất

    col, row, width, height = cv2.boundingRect(largest_rectangle[1])  # Xác định tọa độ của hình chữ nhật

    cv2.drawContours(img, [largest_rectangle[1]], 0, (0, 255, 0), 1)
    show_img('Định vị biển số xe trên ảnh', img)
    cropped_img = img[row:row + height, col:col + width]  # Cắt ảnh bằng tọa độ của hình chữ nhật
    show_img('Biển số xe', cropped_img)
    cropped_img = gray_img[row:row + height, col:col + width]
    return cropped_img


def read_license_plate(img, enable_img):
    global video_mode
    video_mode = not enable_img
    blur = cv2.GaussianBlur(img, (3, 3), 0)  # lam mo anh de giam nhieu va lam gon bien so
    binary_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (3, 3))  # tao kernel de mo rong anh nhi phan de nhan dang bien so
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)  # mo rong
    data = pytesseract.image_to_string(opening, lang='eng',
                                       config='--psm 6')  # su dung pytesseract de nhan dang bien so

    show_img('Bien so la', opening)
    return data


def main():
    global video_mode
    video_mode = False
    img_path = r'asset/images/1.png'
    image = cv2.imread(img_path)
    assert image is not None, 'Khong the doc anh tai duong dan: ' + img_path
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    print('Bien so la:')
    img = find_largest_rectangle(image, True)
    res = read_license_plate(img, True)
    print(res)
    cv2.putText(image, res, (20, 40), cv2.QT_FONT_NORMAL, 1, (255, 255, 255))
    cv2.imshow('Ket qua', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
