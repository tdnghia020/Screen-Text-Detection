# python text_detection_video.py --east frozen_east_text_detection.pb

# import các thư viện cần thiết
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import imutils
import time
import cv2

# Định nghĩa hàm decode_predictions, hàm này dùng để trích xuất:
# 1. Bounding box của 1 vùng văn bản
# 2. Xác suất phát hiện vùng văn bản
def decode_predictions(scores, geometry):

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	
	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]


		for x in range(0, numCols):

			if scoresData[x] < args["min_confidence"]:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)


			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

		
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)


			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])


	return (rects, confidences)

# xây dựng trình phân tích cú pháp đối số và phân tích các đối số
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, # Ngưỡng xác suất để xác định văn bản
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, # Kích thức chiều rộng, phải là bội của 32
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320, # Kích thước chiều cao phải là bội của 32
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# Khởi tạo kích thước khung ban đầu, kích thước khung mới
# và tỉ lệ giữa các kích thước
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# Định nghĩa tên 2 lớp output của EAST detector model: 
# Đầu tiên là output probabilities 
# Cái thứ 2 có thể được sử dụng để lấy tọa độ bounding box giới hạn của văn bản
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# Nếu đường dẫn video không được cung cấp sẽ tham chiếu đến webcam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# Tham chiếu đến tệp video
else:
	vs = cv2.VideoCapture(args["video"])

# khởi động công cụ ước tính  FPS
fps = FPS().start()

# Lặp lại các khung hình từ luồng video
while True:
	# Lấy khung hiện tại, sau đó xử lí nếu đang dùng VideoStream hoặc VideoCapture 
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# kiểm tra đã đến cuối luồng chưa
	if frame is None:
		break

	# thay đổi kích thước khung hình, duy trì tỉ lệ khung hình
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	# nếu kishc thước khung hình là None, 
	# vẫn cần tính toán tỉ lệ kích thước khung hình cũ với kích thước khung hình mới
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# Thay đổi kích thước khung hình, lần này bỏ qua tỷ lệ khung hình
	frame = cv2.resize(frame, (newW, newH))

	#phát hiện vùng văn bản bằng EAST thông qua việc tạo một đốm màu và truyền nó qua mạng
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# giải mã dự đoán (áp dụng NMS)
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# lặp lại các bounding box và vẽ chúng trên khung
	for (startX, startY, endX, endY) in boxes:
		# chia tỉ lệ tọa độ bounding box giới hạn dựa trên tỉ lệ tương ứng
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		roi = orig[startY:endY, startX:endX]
		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)
		# vẽ bounding box trên khung và đẩy text nhận diện được lên
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		cv2.putText(orig, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
	# update bộ đếm FPS
	fps.update()


	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF

	# dừng vòng lặp khi press 'q'
	if key == ord("q"):
		break

# dừng timer và hiển thị thông tin FPS
fps.stop()
print("[INFO] elasped time: {:.02f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.02f}".format(fps.fps()))

#  nếu dùng webcam, hiển thị con trỏ 
if not args.get("video", False):
	vs.stop()

# nếu không, giải phóng con trỏ
else:
	vs.release()


cv2.destroyAllWindows()