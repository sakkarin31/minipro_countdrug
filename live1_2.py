import cv2
import numpy as np

def count_pills_edge(frame):
    # แปลงเป็น Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # เพิ่มความคมชัดด้วย CLAHE มาคไว้เลยเเชทคำสั่งนี้ดีมาก
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Threshold แบบ Binary
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Morphological Closing เติมขอบวัตถุ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ใช้ Canny Edge Detection เพื่อแสดง Edges
    edges = cv2.Canny(closing, 50, 150)

    # หา Contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # กรองขนาด Contours และรวม Contours ที่ทับซ้อน
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 5000:  # กรองขนาด Contours ให้ใกล้เคียงเม็ดยา
            filtered_contours.append(cnt)

    # วาด Bounding Box รอบวัตถุที่ผ่านการกรอง
    result = frame.copy()
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # แสดงจำนวนเม็ดยา
    num_pills = len(filtered_contours)
    cv2.putText(result, f"Pills: {num_pills}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return result, edges

# เปิดการเชื่อมต่อวิดีโอ
cap = cv2.VideoCapture("http://192.168.0.101:8080/video")

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
    processed_frame, edges = count_pills_edge(frame)
    
    # แสดงภาพ
    cv2.imshow("pill on mobile", processed_frame)
    cv2.imshow("Edges", edges)
    
    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) == ord("q"):
        break

# ปิดการเชื่อมต่อและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
