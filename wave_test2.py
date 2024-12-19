import cv2
import numpy as np

def analyze_pills(image_path):
    # อ่านภาพจากไฟล์
    image = cv2.imread(image_path)
    original = image.copy()

    # แปลงภาพเป็น Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ใช้ CLAHE เพื่อปรับแสงและคอนทราสต์
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ใช้ Gaussian Blur เพื่อลด Noise
    blurred = cv2.GaussianBlur(enhanced, (21, 21), 0)

    # ใช้ Adaptive Thresholding เพื่อแยกขอบเม็ดยา
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 15, 4)

    # ทำ Morphological Closing เพื่อปิดช่องว่างเล็กๆ ที่ขอบ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    # หา Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ประเภทของยา
    round_pills = 0
    oval_pills = 0
    damaged_pills = 0

#hiiiiiiii
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # กรองขนาดที่เล็กเกินไป
        if area < 500:
            continue

        # คำนวณความกลม (circularity)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

        # คำนวณอัตราส่วน (Aspect Ratio)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0

        # ตรวจสอบรูปร่าง
        if circularity > 0.75 and 0.9 < aspect_ratio < 1.1:  # เม็ดกลม
            round_pills += 1
            cv2.drawContours(original, [cnt], -1, (0, 255, 0), 2)  # วาดเป็นสีเขียว
        elif 1.5 < aspect_ratio < 3.0:  # เม็ดวงรีหรือแคปซูล
            oval_pills += 1
            cv2.drawContours(original, [cnt], -1, (255, 0, 0), 2)  # วาดเป็นสีน้ำเงิน
        else:  # เม็ดยาเสียหายหรือผิดปกติ
            damaged_pills += 1
            cv2.drawContours(original, [cnt], -1, (0, 0, 255), 2)  # วาดเป็นสีแดง

    # แสดงผลลัพธ์
    print("ผลการวิเคราะห์เม็ดยา:")
    if round_pills > 0:
        print(f"- ยาเม็ดกลม: {round_pills} เม็ด")
    if oval_pills > 0:
        print(f"- ยาเม็ดวงรีหรือแคปซูล: {oval_pills} เม็ด")
    if damaged_pills > 0:
        print(f"- เม็ดยาเสียหาย: {damaged_pills} เม็ด")
    if round_pills == 0 and oval_pills == 0:
        print("ไม่มีเม็ดยาที่ตรวจพบ")
    if damaged_pills == 0:
        print("แสดงว่าไม่มียาเสียหาย")

    # แสดงภาพ
    cv2.imshow("Analyzed Pills", original)
    cv2.imshow("Edges", adaptive_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# เรียกใช้ฟังก์ชัน
image_path = r"C:\Users\focus\minipro_countdrug\original (1).jpg"  # เปลี่ยนเป็นพาธของภาพ
analyze_pills(image_path)
