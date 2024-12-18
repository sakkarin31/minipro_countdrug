import cv2
import numpy as np

# ฟังก์ชันสำหรับตรวจจับและนับเม็ดยาโดยใช้ Edge Detection
def count_pills_edge(frame):
    # สร้างสำเนาเพื่อแสดงผล
    original = frame.copy()
    
    # แปลงเป็น Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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
    
    # นับจำนวนเม็ดยา
    num_pills = len(pill_contours)
    
    # ใส่ข้อความแสดงจำนวนเม็ดยา
    cv2.putText(original, f"Pills: {num_pills}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return original, edges

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
    cv2.imshow("Live Feed with Pill Detection", processed_frame)
    cv2.imshow("Edges", edges)
    
    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) == ord("q"):
        break

# ปิดการเชื่อมต่อและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture("http://192.168.0.101:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from the video stream.")
        break
    processed_frame, edges = count_pills_edge(frame)
    

    cv2.imshow("Live Feed with Pill Detection", processed_frame)
    cv2.imshow("Edges", edges)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
