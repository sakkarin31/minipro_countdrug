import cv2
import numpy as np

# ฟังก์ชันสำหรับนับเม็ดยา
def count_pills_with_watershed(image_path):
    # โหลดภาพ
    image = cv2.imread(image_path)
    original = image.copy()

    # แปลงภาพเป็นสีเทา
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # การเบลอเพื่อลดสัญญาณรบกวน
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # ใช้ Threshold แยกวัตถุจากพื้นหลัง
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ทำ Morphological Closing เพื่อลบรอยรั่วเล็ก ๆ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # คำนวณ Distance Transform
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    
    # Normalize ค่า Distance Transform
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    # Threshold เพื่อแยกจุดศูนย์กลางของเม็ดยา
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # หาพื้นที่ที่ไม่แน่ใจ (Background)
    sure_bg = cv2.dilate(closed, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker สำหรับ Watershed Algorithm
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # ใช้ Watershed Algorithm
    markers = cv2.watershed(image, markers)

    # วาด Contour ของเม็ดยา
    pill_count = 0
    for marker_id in np.unique(markers):
        if marker_id <= 1:  # ข้ามพื้นหลังและส่วนที่ไม่แน่ใจ
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker_id] = 255

        # ค้นหา Contour ของเม็ดยา
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:  # กรอง Contour ที่เล็กเกินไป
                pill_count += 1
                cv2.drawContours(original, [cnt], -1, (0, 255, 0), 2)

    # แสดงผล
    print(f"จำนวนเม็ดยาที่ตรวจพบ: {pill_count}")
    cv2.imshow("Result", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# เรียกใช้ฟังก์ชัน
count_pills_with_watershed(r"C:\Users\sakka\Downloads\drugtest5.jpg")
