import cv2
import numpy as np

def count_pills_edge(frame):
    # แปลงเป็น Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # เพิ่มความคมชัดด้วย CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Threshold แบบ Binary
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ใช้ Canny Edge Detection
    edges = cv2.Canny(closing, 50, 150)

    # ใช้ภาพต้นฉบับเป็น result โดยไม่มีการวาดกรอบ
    result = frame.copy()

    # คืนค่า result และ edges
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
