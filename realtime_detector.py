import cv2
from ultralytics import YOLO
import time # FPS calculate 

# Pre-trained YOLOv8n model  load 
model = YOLO('yolov8n.pt')

#  default webcam  open 
cap = cv2.VideoCapture(0)

# Camera not open give a  error
if not cap.isOpened():
    print("Error: cannot open the camara!")
    exit()

# FPS calculate  variables (Optional)
prev_frame_time = 0
new_frame_time = 0

print("Camara is open,now can identify objects.")
print("To close the window press 'q'...")

# Loop started - camera takes the  frames
while True:
    # Camera get the  frame  (image ) 
    success, frame = cap.read()

    # cannot get the Frame,stop the loop.
    if not success:
        print("Error: Cannot get the frames from camara!")
        break

    # --- Object Detection ---
    #  frame input into YOLO model and objects detect..
    # stream=True 
    results = model(frame, stream=True)

    # --- FPS Calculation (Optional) ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # print(f"FPS: {int(fps)}") # Console FPS.

    # --- Show results ---
    for r in results:
        # Ultralytics library , bounding boxes, labels, confidence scores
        annotated_frame = r.plot() # This function draws the boxes and labels on the frame

        # FPS show over the  frame  (Optional)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        # identify objects frame showw window 
        cv2.imshow("YOLOv8 Real-Time Object Detection", annotated_frame)

    # 'q' key  press close the  loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Close the window,now exit the program.")
        break

# Out of the Loop 
# Release  the camara
cap.release()
# close OpenCV windows
cv2.destroyAllWindows()

print("System is closed succesfully.")