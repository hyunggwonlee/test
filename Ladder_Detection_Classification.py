import torch
import cv2
import numpy as np
import datetime
import time
import shutil
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def classification(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Compute the classification report
    cr = classification_report(y_true, y_pred)
    print("Classification Report:\n", cr)


    # Compute the ROC AUC score
    auc_score = roc_auc_score(y_true, y_pred)

    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')


def detect_ladder(image, model):
    start_time = datetime.datetime.now()
    results = model(image)
    global_behaviour = "Safe"
    worker = []
    ladder = []
    reason = []

    finalStatus = ""
    if np.shape(results.xyxy[0].cpu().numpy())[0] > 0:
        for (x0, y0, x1, y1, confi, clas) in results.xyxy[0].cpu().numpy():

            if confi > 0.35:
                # print(x0, y0, x1, y1, confi, clas)
                box = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                box2 = [int(x0), int(y0), int(x1), int(y1)]
                if int(clas) == 0:
                    cv2.rectangle(image, box, (0, 128, 0), 2)
                    cv2.putText(image, "ladder_with_outriggers {:.2f}".format(confi), (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 128, 0), 2)
                    ladder.append(box2)

                # elif int(clas) == 1 or int(clas) == 0 and False:
                elif int(clas) == 1:

                    global_behaviour = "UnSafe"
                    if "Ladder without Outtrigger" not in reason:
                        reason.append("Ladder without Outtrigger")
                        ladder.append(box2)
                        cv2.rectangle(image, box, (255, 0, 0), 2)
                        cv2.putText(image, "ladder_without_outriggers {:.2f}".format(confi), (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2, (255, 0, 0), 2)
                    # Add the box coordinates to the person array
                elif int(clas) == 2:

                    worker.append((box2, box, int(clas)))
                    # Add the box coordinates to the person array
                elif int(clas) == 3:

                    global_behaviour = "UnSafe"
                    if 'Worker Without Helmet' not in reason:
                        reason.append("Worker Without Helmet")
                    worker.append((box2, box, int(clas)))

        worker_height = 0
        co_worker = 0
        worker_with_height = []
        worker_without_height = []
        highest_height = -1
        if len(ladder) == 0:
            for worker_box, worker_box_bounding, worker_class in worker:
                if int(worker_class) == 2:
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "worker_with_helmet {:.2f}".format(confi), (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2)

                    # Add the box coordinates to the person array
                elif int(worker_class) == 3:
                    cv2.rectangle(image, box, (255, 0, 0), 2)
                    cv2.putText(image, "worker_without_helmet {:.2f}".format(confi), (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 0, 0), 2)
        for ladder_box in ladder:
            for worker_box, worker_box_bounding, worker_class in worker:  # box2 = [int(x0), int(y0), int(x1), int(y1)]
                if ((ladder_box[0] <= worker_box[0]) and (worker_box[0] <= ladder_box[2])) or ((ladder_box[0] <= worker_box[2]) and (worker_box[2] <= ladder_box[2])):
                    ladder_height_in_px = ladder_box[3] - ladder_box[1]
                    worker_height_in_px = ladder_box[3] - worker_box[3]
                    ladder_height_in_percentage = ladder_height_in_px * 2.0 / 100.0
                    worker_height_in_percentage = (worker_height_in_px / ladder_height_in_px) * 100.0
                    worker_height = (worker_height_in_percentage / 100.0) * 2.0
                    worker_height = round(worker_height, 2)
                    if highest_height < worker_height:
                        highest_height = worker_height
                        worker_with_height.insert(0, (worker_box, worker_box_bounding, worker_class, worker_height))
                    else:
                        worker_with_height.append((worker_box, worker_box_bounding, worker_class, worker_height))
                else:
                    worker_without_height.append((worker_box, worker_box_bounding, worker_class))

        for worker_box, worker_box_bounding, worker_class in worker_without_height:

            if int(worker_class) == 2:
                cv2.rectangle(image, box, (0, 255, 0), 2)
                cv2.putText(image, "worker_with_helmet {:.2f}".format(confi), (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

                # Add the box coordinates to the person array
            elif int(worker_class) == 3:
                cv2.rectangle(image, box, (255, 0, 0), 2)
                cv2.putText(image, "worker_without_helmet {:.2f}".format(confi), (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 0, 0), 2)
        for x in range(0, len(worker_with_height)):

            worker_box, worker_box_bounding, worker_class, height_ladder_worker = worker_with_height[x]
            if x == 0:
                if int(worker_class) == 2:
                    cv2.rectangle(image, worker_box_bounding, (0, 255, 0), 2)
                    cv2.putText(image, "worker_with_helmet {:.2f}".format(confi), (worker_box_bounding[0], worker_box_bounding[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 128, 0), 2)

                    # Add the box coordinates to the person array
                elif int(worker_class) == 3:
                    cv2.rectangle(image, worker_box_bounding, (255, 0, 0), 2)
                    cv2.putText(image, "worker_without_helmet {:.2f}".format(confi), (worker_box_bounding[0], worker_box_bounding[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 0, 0), 2)
            else:
                co_worker = co_worker + 1
                if int(worker_class) == 2:
                    cv2.rectangle(image, worker_box_bounding, (219, 252, 3), 2)
                    cv2.putText(image, "CO-worker_with_helmet {:.2f}".format(confi), (worker_box_bounding[0], worker_box_bounding[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (219, 252, 3), 2)

                    # Add the box coordinates to the person array
                elif int(worker_class) == 3:
                    cv2.rectangle(image, worker_box_bounding, (252, 3, 219), 2)
                    cv2.putText(image, "CO-worker_without_helmet {:.2f}".format(confi), (worker_box_bounding[0], worker_box_bounding[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (252, 3, 219), 2)

        worker_height = highest_height
        if highest_height >= 1.2:
            reason.append(f"Unsafe height : {worker_height} m")

        if highest_height >= 1.2 and co_worker == 0:
            global_behaviour = "UnSafe"
        else:
            if worker_height <= 0:
                worker_height = 0
        finalStatus = global_behaviour
        if finalStatus != "Safe":
            if worker_height != 0 and worker_height < 1.2:
                cv2.putText(image, f"{finalStatus} : Height : {worker_height} m", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),
                            2)
            else:
                cv2.putText(image, f"{finalStatus}  ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            increment = 90
            for rea in reason:
                cv2.putText(image, f"{rea}", (50, increment), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                increment = increment + 40

        else:
            cv2.putText(image, f"{finalStatus}  Height : {worker_height} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    end_time = datetime.datetime.now()
    inference_time = (end_time - start_time).total_seconds()

    if finalStatus == "UnSafe":
        return image, 0, inference_time
    else:
        return image, 1, inference_time


if __name__ == "__main__":
    total_inference_time = 0.0
    ground_truth = []
    prediction = []

    Ladder_model = torch.hub.load('WongKinYiu/yolov7', 'custom', './Ladder.pt', force_reload=False)

    print("Loading...")



    safe_folder_path = "./Safe"
    unsafe_folder_path = "./Unsafe"
    output_folder_path = "./Predictions"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Folder '{output_folder_path}' created.")
    else:
        shutil.rmtree(output_folder_path)
        os.makedirs(output_folder_path)
        print(f"Old folder deleted and recreate the folder '{output_folder_path}' .")

    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(safe_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(safe_folder_path, image_file)
        image = cv2.imread(image_path)
        ground_truth.append(1)
        mobile_scaff_result_image, status, inference_time = detect_ladder(image, Ladder_model)
        total_inference_time += inference_time
        prediction.append(status)
        output_path = os.path.join(output_folder_path, image_file  + ".jpg")
        cv2.imwrite(output_path, mobile_scaff_result_image)

    # Get a list of image file names in the folder
    unsafe_image_files = [f for f in os.listdir(unsafe_folder_path) if
                          f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for unsafe_image_file in unsafe_image_files:
        # Read the image
        image_path = os.path.join(unsafe_folder_path, unsafe_image_file)
        image = cv2.imread(image_path)
        ground_truth.append(0)
        mobile_scaff_result_image, status, inference_time = detect_ladder(image, weld_cut_model)
        total_inference_time += inference_time
        prediction.append(status)
        output_path = os.path.join(output_folder_path, unsafe_image_file + ".jpg")
        cv2.imwrite(output_path, mobile_scaff_result_image)
    num_images = len(image_files) + len(unsafe_image_files)
    avg_inference_time = total_inference_time / num_images if num_images > 0 else 0.0
    print(f"Average Inference Time: {avg_inference_time} seconds")
    print("FPS = ", 1 / avg_inference_time)
    classification(ground_truth, prediction)

