import cv2
import numpy as np

# ฟังก์ชันสำหรับตรวจจับและวิเคราะห์เม็ดยา
def analyze_pills(image_path):
    # อ่านภาพจากไฟล์
    image = cv2.imread(image_path)
    original = image.copy()
    
    # แปลงเป็น Grayscale และเบลอเพื่อลด Noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 7), 0)
    
    # ใช้ Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # ทำ Morphological Closing เพื่อปิดรูที่ขอบ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # หา Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ประเภทของยา
    round_pills = 0
    oval_pills = 0
    damaged_pills = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # กรองขนาดที่เล็กเกินไป
        if area < 300:
            continue
        
        # คำนวณความกลม (circularity)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
        
        # คำนวณอัตราส่วน (Aspect Ratio)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # ตรวจสอบรูปร่าง
        if circularity > 0.8:  # เม็ดกลม
            round_pills += 1
            cv2.drawContours(original, [cnt], -1, (0, 255, 0), 2)  # วาดเป็นสีเขียว
        elif 0.4 < aspect_ratio < 1 or 1.5 < aspect_ratio < 2.5:  # เม็ดวงรีหรือแคปซูล
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
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# เรียกใช้ฟังก์ชัน
image_path = r"C:\Users\sakka\Downloads\drugtest.jpg"  # เปลี่ยนเป็นพาธของภาพ
analyze_pills(image_path)

#test

