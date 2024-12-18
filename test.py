import cv2
import numpy as np

# ฟังก์ชันสำหรับตรวจจับและนับเม็ดยาโดยใช้ Edge Detection
def count_pills_edge(image_path):
    # อ่านภาพจากไฟล์
    image = cv2.imread(image_path)
    original = image.copy()
    
    # แปลงเป็น Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ใช้ GaussianBlur เพื่อลด Noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # ใช้ Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # ทำการปิดขอบด้วย Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # หา Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # กรองขนาด Contours ที่ไม่ใช่เม็ดยา
    pill_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    
    # วาดกรอบรอบเม็ดยา
    for cnt in pill_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # แสดงผลลัพธ์
    num_pills = len(pill_contours)
    print(f"จำนวนเม็ดยาที่ตรวจพบ: {num_pills}")
    cv2.imshow("Original Image with Pills", original)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return num_pills

# เรียกใช้ฟังก์ชัน
image_path = r"C:\Users\sakka\Downloads\drugtest.jpg"  # เปลี่ยนเป็นพาธของภาพที่ต้องการ
count_pills_edge(image_path)
