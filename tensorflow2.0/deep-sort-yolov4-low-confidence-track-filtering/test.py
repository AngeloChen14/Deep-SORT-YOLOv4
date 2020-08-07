import cv2
file_path = 'reid-wide1'
video_capture = cv2.VideoCapture(file_path+".mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("cut.avi", fourcc, 12, (520, 342))
num = 0
while True:
    ret, frame = video_capture.read()
    num += 1 
    if ret != True:
        break
    if num > 12 and num < 72:
        cv2.waitKey(1)
        out.write(frame)
        print('!!!')
    if num > 162:
        cv2.waitKey(1)
        out.write(frame)
        print('!!!')
print('OK!')
out.release()
video_capture.release()