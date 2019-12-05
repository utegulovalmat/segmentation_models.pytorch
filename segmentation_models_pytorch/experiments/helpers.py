import os
import base64
import requests
import matplotlib.pyplot as plt
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

plt.rcParams["figure.figsize"] = (7, 7)

logger = None


class NoMatchingModelException(Exception):
    pass


def get_volume_paths(input_dir: str):
    """Get paths to volumes

    :param input_dir: directory with MRI volumes
    :return: sorted list of paths to masks and volume images
    """
    fns = []
    mask_fns = []
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if "nrrd" in filename and "seg" not in filename:
                if "label.nrrd" in filename:
                    mask_fns.append(os.path.join(dirname, filename))
                else:
                    fns.append(os.path.join(dirname, filename))
    mask_fns = sorted(mask_fns)
    fns = sorted(fns)
    return mask_fns, fns


def get_volume_fn_from_mask_fn(mask_filename):
    """Extract volume path from mask path"""
    return mask_filename.replace("S-label", "")


def save_sample_image(image, mask, output_dir):
    fig = plt.figure(figsize=(7, 7))
    rows, columns = 1, 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask)
    # plt.show()
    plt.savefig(fname=output_dir + "/input_sample.png")


def format_history(train_history, valid_history):
    history = {
        "loss": [],
        "iou_score": [],
        "f-score": [],
        "val_loss": [],
        "val_iou_score": [],
        "val_f-score": [],
    }
    for train_log, valid_log in zip(train_history, valid_history):
        history["loss"].append(train_log["dice_loss"])
        history["iou_score"].append(train_log["iou_score"])
        history["f-score"].append(train_log["fscore"])
        history["val_loss"].append(valid_log["dice_loss"])
        history["val_iou_score"].append(valid_log["iou_score"])
        history["val_f-score"].append(valid_log["fscore"])
    return history


def format_test_result_metrics(best_test_row):
    epoch = "epoch: XX"
    loss = "loss: {:.5}".format(best_test_row["dice_loss"])
    iou = "iou: {:.5}".format(best_test_row["iou_score"])
    dice = "dice: {:.5}".format(best_test_row["fscore"])
    return " | ".join([epoch, loss, iou, dice])


def get_best_metrics(history):
    best_train_loss = 1e10
    best_train_row = ""
    for idx, (loss, iou, dice) in enumerate(
        zip(history["loss"], history["iou_score"], history["f-score"])
    ):
        if loss < best_train_loss:
            best_train_loss = loss
            epoch = "epoch: " + str(idx)
            loss = "loss: {:.5}".format(loss)
            iou = "iou: {:.5}".format(iou)
            dice = "dice: {:.5}".format(dice)
            best_train_row = " ".join([epoch, loss, iou, dice])

    best_valid_loss = 1e10
    best_valid_row = ""
    for idx, (loss, iou, dice) in enumerate(
        zip(history["val_loss"], history["val_iou_score"], history["val_f-score"])
    ):
        if loss < best_valid_loss:
            best_valid_loss = loss
            epoch = "epoch: " + str(idx)
            loss = "loss: {:.5}".format(loss)
            iou = "iou: {:.5}".format(iou)
            dice = "dice: {:.5}".format(dice)
            best_valid_row = " ".join([epoch, loss, iou, dice])

    print("Best train |", best_train_row)
    print("Best valid |", best_valid_row)
    return best_train_row, best_valid_row


def plot_graphs(history, output_dir):
    print("plot_graphs", history["iou_score"])
    print("plot_graphs", history["val_iou_score"])
    # Plot training & validation iou_score values
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.plot(history["iou_score"])
    plt.plot(history["val_iou_score"])
    plt.title("Model IoU score")
    plt.ylabel("IoU score")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    # Plot training & validation dice values
    plt.subplot(132)
    plt.plot(history["f-score"])
    plt.plot(history["val_f-score"])
    plt.title("Model F-score")
    plt.ylabel("F-score")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    # Plot training & validation loss values
    plt.subplot(133)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    # plt.show()
    plt.savefig(output_dir + "/graphs.png")


def get_datetime_str():
    from datetime import datetime

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M-%S")
    return date + "-" + time


def send_email(title, message):
    """
    import base64
    encoded = base64.b64encode(b'...')
    data = base64.b64decode(encoded)
    """
    api = os.environ.get("SENDGRID_API_KEY")
    to = base64.b64decode(b"QWxtYXQgPHV0ZWd1bG92QHVuaS1rb2JsZW56LmRlPg==").decode()
    message = Mail(
        from_email=to,
        to_emails=to,
        subject="Model training: " + title,
        html_content=message,
    )
    try:
        sg = SendGridAPIClient(api)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))
