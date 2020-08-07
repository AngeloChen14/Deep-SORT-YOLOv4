#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker, cross_tracker_match,ROI_target_match
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.25
    max_cross_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.5
    frame_rate = 12

    file_path = 'reid-wide2'
    file_path2 = 'reid-long2'

    show_detections = False
    writeVideo_flag = False
    asyncVideo_flag = False
    beta_calulate_flag = False
    predict_ns_flag = False
    predict_time = 0.5
    multi_camera_flag = True 

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric,frame_rate=frame_rate)
    
    if multi_camera_flag:
        metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker2 = Tracker(metric2,frame_rate=frame_rate)

    if asyncVideo_flag:
        video_capture = VideoCaptureAsync(file_path+".mp4")
    else:
        video_capture = cv2.VideoCapture(file_path+".mp4")
    
    if multi_camera_flag:
        video_capture2 = cv2.VideoCapture(file_path2+".mp4")

    if asyncVideo_flag:
        video_capture.start()

    if asyncVideo_flag:
        w = int(video_capture.cap.get(3))
        h = int(video_capture.cap.get(4))
    else:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))

    if writeVideo_flag:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path2+"-output.avi", fourcc, frame_rate, (w, h))
        frame_index = -1

        if multi_camera_flag:
            out2 = cv2.VideoWriter(file_path+"-output2.avi", fourcc, frame_rate, (w, h))

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    alpha = np.arctan((326+ (369-326)*(250-110)/(300-110)-w/2)/w*6.4/3.6)* 180 /np.pi
    beta = 0
    beta_last = 0
    x_temp = 0
    track_angle =np.array([])
    x_axis =np.array([])
    x_predict = np.array([])
    x_current = np.array([])
    x_kalman = np.array([])
    matches_id = []
    alert_mode_flag = False

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        features = encoder(frame, boxes)
        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                      zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.cls for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        if predict_ns_flag:
            tracker.predict_ns(predict_time)
        
        if alert_mode_flag:
            if len(tracker.tracks) > p:
                ret =  tracker.tracks[track1_idx].is_confirmed()
                if not ret:
                    alert_mode_flag = False
            else:
                alert_mode_flag = False

        if not alert_mode_flag:
            for det in detections:
                bbox = det.to_tlbr()
                if show_detections and len(classes) > 0:
                    det_cls = det.cls
                    score = "%.2f" % (det.confidence * 100) + "%"
                    cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                                1e-3 * frame.shape[0], (0, 255, 0), 1)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 5:
                    continue
                bbox = track.to_tlbr()

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)
                
                if predict_ns_flag:
                    if track.x_predict > 0 and track.x_predict < w :
                        cv2.circle(frame, (track.x_predict, int(h/2)), 15, (0, 0, 255), -1)
                    x_temp += 1 / frame_rate
                    x_axis = np.append(x_axis, x_temp)
                    x_predict = np.append(x_predict, track.x_predict)
                    x_current = np.append(x_current, track.x_current)
                    x_kalman =  np.append(x_kalman,  track.mean[0])

                if beta_calulate_flag:
                    beta = np.arctan((track.mean[0]-w/2)/w*6.4/16)* 180 / np.pi
        else:              
            bbox = tracker.tracks[track1_idx].to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                        1e-3 * frame.shape[0], (0, 255, 0), 1)

            # if not show_detections:
            #     track_cls = track.cls
            #     adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
            #     cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0], (0, 255, 0),
            #                 1)
            #     cv2.putText(frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
            #                 1e-3 * frame.shape[0], (0, 255, 0), 1)

        if beta_calulate_flag:
            if np.isclose(beta,beta_last) and abs(beta) > 7:
                    beta = np.sign(beta)*np.arctan(3.2/16)* 180 / np.pi
            cv2.putText(frame, "Beta_angle: " + '{:4.2f}'.format(beta), (20,20), 0,
                    1e-3 * frame.shape[0], (0, 0, 255), 1)
            beta_last = beta 
            x_temp += 1 / frame_rate
            x_axis = np.append(x_axis, x_temp)
            track_angle = np.append(track_angle, alpha+beta)

        cv2.imshow('camera1', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        if multi_camera_flag:
            ret, frame = video_capture2.read()  # frame shape 640*480*3
            if ret != True:
                break

            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            boxes, confidence, classes = yolo.detect_image(image)

            features = encoder(frame, boxes)
            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                        zip(boxes, confidence, classes, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.cls for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker2.predict()
            tracker2.update(detections)
            matches, unmatched_tracks_1, unmatched_tracks_2 = cross_tracker_match(tracker, tracker2, max_cross_cosine_distance)
            matches_id.clear()
            for track_1_idx, track_2_idx in matches:
                matches_id.append((tracker.tracks[track_1_idx].track_id,tracker2.tracks[track_2_idx].track_id))
            print("Matches:", matches_id)
            
            if alert_mode_flag:
                if len(tracker2.tracks) > track2_idx:
                    ret =  tracker2.tracks[track2_idx].is_confirmed()
                    if not ret:
                        alert_mode_flag = False
                else:
                    alert_mode_flag = False

            if not alert_mode_flag:
                for det in detections:
                    bbox = det.to_tlbr()
                    if show_detections and len(classes) > 0:
                        det_cls = det.cls
                        score = "%.2f" % (det.confidence * 100) + "%"
                        cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                                    1e-3 * frame.shape[0], (0, 255, 0), 1)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

                for track in tracker2.tracks:
                    if not track.is_confirmed() or track.time_since_update > 5:
                        continue
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                                1e-3 * frame.shape[0], (0, 255, 0), 1)
                
                if cv2.waitKey(1) & 0xFF == ord('t'):
                    roi = cv2.selectROI("Select Target",frame)
                    Roi = [Detection(roi, 0, None, 0)]
                    ret, track2_idx = ROI_target_match(tracker2, Roi)
                    cv2.destroyWindow("Select Target")
                    
                    if ret:
                        t_list = [i for (i,j) in matches if j == track2_idx]
                        if t_list != []:
                            track1_idx = t_list[0]
                            alert_mode_flag = True
            else:
                bbox = tracker2.tracks[track2_idx].to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

            cv2.imshow('camera2', frame)

            if writeVideo_flag:
                # save a frame
                out2.write(frame)
            
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # if not asyncVideo_flag:
        #     fps = (fps + (1. / (time.time() - t1))) / 2
        #     print("FPS = %.2f" % (fps))
        fps_imutils.update()

    fps_imutils.stop()
    print('imutils FPS: {:.2f}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()
        if multi_camera_flag:
            out2.release()

    if predict_ns_flag:
        np.save("x_axis.npy",x_axis)
        np.save("x_predict.npy",x_predict)
        np.save("x_current.npy",x_current) 
        np.save("x_kalman.npy",x_kalman) 
        plt.plot(x_axis, x_current, label='X_Measured')
        plt.plot(x_axis, x_kalman, label='X_Filtered')
        plt.plot(x_axis, x_predict, label='X_Predicted')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Pixel')
        plt.title('Motion Prediction')
        plt.show()

    if beta_calulate_flag:
        np.save("x_axis.npy",x_axis)
        np.save("track_angle.npy",track_angle)
        plt.plot(x_axis, track_angle)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('Target Angle')
        plt.show()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
