import cv2
import numpy as np

# ฟังก์ชันสำหรับตรวจจับและวิเคราะห์เม็ดยา
def analyze_pills(frame):
    # แปลงเป็น Grayscale และเบลอเพื่อลด Noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)  # วาดเป็นสีเขียว
        elif 0.4 < aspect_ratio < 1 or 1.5 < aspect_ratio < 2.5:  # เม็ดวงรีหรือแคปซูล
            oval_pills += 1
            cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)  # วาดเป็นสีน้ำเงิน
        else:  # เม็ดยาเสียหายหรือผิดปกติ
            damaged_pills += 1
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)  # วาดเป็นสีแดง
    
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
    cv2.imshow("Analyzed Pills", frame)
    cv2.imshow("Edges", edges)

# เรียกใช้ฟังก์ชันกับสตรีมวิดีโอ
cap = cv2.VideoCapture("http://10.78.51.72:8080/video")

if not cap.isOpened():
    print("Error: Unable to access the video stream.")
    exit()

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from the video stream.")
        break

    # ตรวจจับและแสดงผล
    processed_frame = analyze_pills(frame)
    

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) == ord("q"):
        break

# ปิดการเชื่อมต่อและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows() 