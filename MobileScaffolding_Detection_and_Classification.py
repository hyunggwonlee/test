import torch
import cv2
import numpy as np
import datetime
import os
import shutil
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

def detect_mobile_scaffolding(image, model):
    start_time = datetime.datetime.now()
    results = model(image)

    worker_with_helmet = []
    worker_without_helmet = []
    mobile_scaffold_outrigger = []
    finalStatus = "Safe"
    finalMessage = []
    total_no_person_on_scaffolding = 0
    if np.shape(results.xyxy[0].cpu().numpy())[0] > 0:
        for (x0, y0, x1, y1, confi, clas) in results.xyxy[0].cpu().numpy():
            if confi > 0.5:
                box = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                box2 = [int(x0), int(y0), int(x1), int(y1)]
                if int(clas) == 0:
                    cv2.rectangle(image, box, (0, 0, 255), 2)
                    cv2.putText(image, "Missing Guardrail {:.2f}".format(confi), (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    finalStatus = "UnSafe"
                    finalMessage.append("Missing Guardrail")
                elif int(clas) == 1:
                    cv2.rectangle(image, box, (0, 0, 255), 2)
                    cv2.putText(image, "mobile_scaffold_no_outtrigger {:.2f}".format(confi), (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    finalStatus = "UnSafe"
                    finalMessage.append("No outrigger")
                    mobile_scaffold_outrigger.append(box2)
                elif int(clas) == 2:
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "mobile_scaffold_outtrigger {:.2f}".format(confi), (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 128, 0), 2)
                    mobile_scaffold_outrigger.append(box2)
                elif int(clas) == 3 :
                    cv2.rectangle(image, box, (0, 128, 0), 2)
                    cv2.putText(image, "worker_with_helmet {:.2f}".format(confi), (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 128, 0), 2)
                    worker_with_helmet.append(box2)
                elif int(clas) == 4:
                    cv2.rectangle(image, box, (0, 0, 255), 2)
                    cv2.putText(image, "worker_without_helmet {:.2f}".format(confi), (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    finalStatus = "UnSafe"
                    if "Missing Helmet" not in finalMessage:
                        finalMessage.append("Missing Helmet")

        for scaff_box in mobile_scaffold_outrigger:
            for per_box in worker_with_helmet:
                if int(scaff_box[0]) < int(per_box[0]) and int(per_box[2]) < int(scaff_box[2]):
                    top_y_scaffolding = scaff_box[1]
                    bottom_y_scaffolding = scaff_box[3]
                    center_y_scaffolding = int((top_y_scaffolding + bottom_y_scaffolding) / 2)
                    if center_y_scaffolding >= per_box[3] - 20:
                        total_no_person_on_scaffolding = total_no_person_on_scaffolding + 1

        if total_no_person_on_scaffolding > 2:
            finalStatus = "UnSafe"
            finalMessage.append(f"{total_no_person_on_scaffolding} Worker")

        if finalStatus != "Safe":
            cv2.putText(image, f"{finalStatus} : {finalMessage} ", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2)
        else:
            cv2.putText(image, f"{finalStatus} : Safe Behavior ", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0),
                        2)

    end_time = datetime.datetime.now()
    inference_time = (end_time - start_time).total_seconds()
    # print(f"Inference Time: {inference_time} seconds")
    if finalStatus == "UnSafe":
        return image, 0, inference_time
    else:
        return image, 1, inference_time


if __name__ == "__main__":
    total_inference_time = 0.0
    ground_truth = []
    prediction = []

    mobile_scaff_model = torch.hub.load('WongKinYiu/yolov7', 'custom', './mobilescaffolding.pt', force_reload=False)

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
        mobile_scaff_result_image, status, inference_time = detect_mobile_scaffolding(image, mobile_scaff_model)
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
        mobile_scaff_result_image, status, inference_time = detect_mobile_scaffolding(image, mobile_scaff_model)
        total_inference_time += inference_time
        prediction.append(status)
        output_path = os.path.join(output_folder_path, unsafe_image_file + ".jpg")
        cv2.imwrite(output_path, mobile_scaff_result_image)

    num_images = len(image_files) + len(unsafe_image_files)
    avg_inference_time = total_inference_time / num_images if num_images > 0 else 0.0
    print(f"Average Inference Time: {avg_inference_time} seconds")
    print("FPS = ", 1 / avg_inference_time)
    classification(ground_truth, prediction)

    cv2.waitKey(0)
