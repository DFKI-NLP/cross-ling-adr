"""..."""
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import wandb

from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from utils.utils import make_sure_path_exists


def plot_precision_recall_curve(results_list, descriptions, mode=""):
    """Take the gold labels and predicted probas and plot PR curve."""
    tick_spacing = 0.2

    fig, ax = plt.subplots(1, 1)

    class_idx = 1

    # we need those for the "zoom in"
    lim_rec = 1
    lim_prec = 1

    f1_scores = [(i, k.metrics["1"]["f1-score"])
                 for i, k in enumerate(results_list)]

    sorted_indices = sorted(f1_scores, key=lambda k: k[1], reverse=True)

    with open("prec_rec_plots.csv", "a") as write_handle:

        # for i, struct in enumerate(results_list):
        for idx, _ in sorted_indices:

            struct = results_list[idx]

            y_true = np.array(struct.y_true)
            # take only the predictions of one class
            y_pred_class_1 = struct.y_score[:, 1]
            y_pred_class_0 = struct.y_score[:, 0]

            prec_1, rec_1, _ = precision_recall_curve(y_true=y_true,
                                                      probas_pred=y_pred_class_1,
                                                      pos_label=1)
            # prec_0, rec_0, _ = precision_recall_curve(y_true=y_true,
            #                                           probas_pred=y_pred_class_0,
            #                                           pos_label=1)

            if lim_rec > min(rec_1):
                lim_rec = min(rec_1) - 0.01
            # if lim_rec > min(rec_0):
            #     lim_rec = min(rec_0) - 0.01

            if lim_prec > min(prec_1):
                lim_prec = min(prec_1) - 0.01
            # if lim_prec > min(rec_0):
            #     lim_prec = min(prec_0) - 0.01
            f1_1 = struct.metrics["1"]["f1-score"]
            ax.plot(rec_1, prec_1, label=descriptions[idx] + f" (F1 = {f1_1:0.2})")

            write_handle.write(f"{descriptions[idx]}\t{rec_1}\t{prec_1}\n")
            # ax.plot(rec_0, prec_0, label=descriptions[i] + " class 0")

            # print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    # add a "skill baseline" by plotting the ratio of all positives
    # no_skill = len(y_true[y_true == 1]) / len(y_true)
    # ax.plot([0, 1], [no_skill, no_skill],
    #         linestyle='--', label='No Skill')

    # # zoom in
    plt.xlim([lim_rec, 1.01])
    plt.ylim([lim_prec, 1.01])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    # show the legend
    plt.legend(loc="best")
    # axis labels
    plt.xlabel('recall')
    plt.ylabel('precision')
    # set title
    plt.title("Precision-Recall Curve Class 1")
    # show the plot
    # plt.show()
    # log the curve plot
    plot = wandb.Image(plt)
    wandb.log({"precision_recall_curve": plot})
    # save plot
    date = datetime.datetime.now().strftime("%d_%m_%y_%H_%M")
    prc_file_name = f"plots/prec_rec_curve_{mode}_{date}.png"
    make_sure_path_exists("plots/")
    fig.savefig(prc_file_name)


def plot_roc_curve(results_list, descriptions, mode=""):
    """..."""
    fig, ax = plt.subplots(1, 1)

    auc_scores = [(i, k.metrics["auc"])
                  for i, k in enumerate(results_list)]

    sorted_indices = sorted(auc_scores, key=lambda k: k[1], reverse=True)

    with open("roc_plots.csv", "a") as write_handle:

        # for j, struct in enumerate(results_list):
        for idx, _ in sorted_indices:
            struct = results_list[idx]

            y_true = np.array(struct.y_true)
            # take only the predictions of one class
            y_pred_class_1 = struct.y_score[:, 1]

            fpr_1, tpr_1, _ = roc_curve(y_true=y_true,
                                        y_score=y_pred_class_1,
                                        pos_label=1)
            auc_1 = auc(fpr_1, tpr_1)
            auc_score = struct.metrics["auc"]

            lw = 2
            ax.plot(fpr_1, tpr_1,
                    lw=lw,
                    label=f"{descriptions[idx]} (area = {auc_1:0.2f})")

            write_handle.write(f"{descriptions[idx]}\t{fpr_1}\t{tpr_1}\n")

    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Receiver operating characteristic for class 1")
    plt.legend(loc="best")
    # plt.show()
    plot = wandb.Image(plt)
    wandb.log({"roc_curve": plot})

    date = datetime.datetime.now().strftime("%d_%m_%y_%H_%M")
    roc_file_name = f"plots/roc_curve_{mode}_{date}.png"
    make_sure_path_exists("plots/")
    fig.savefig(roc_file_name)
