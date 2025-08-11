import cv2

def draw_label(frame, x, y, w, h, label):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
