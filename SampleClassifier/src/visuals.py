
import seaborn as sn
import matplotlib.pyplot as plt


class Visuals:

    LABELS = [
        "brst", "ctrl", "lung",
        "ovrn", "prst", "skin", "tyrd"
    ]

    # --- Learning Curves ---
    @classmethod
    def plot_learning_curve(cls, Eval, scorer='accuracy', title='Learning Curves', scaled_y=False):

        inputs = Eval.calc_inputs  # i, name, model, y_pred, y_test

        n = len(inputs)
        fig, ax = plt.subplots(n//3, 3)

        out = []
        for inp in inputs:
            i = inp['i']  # index of the model
            name = inp['name']  # name of the model

            # calculate learning curves
            train_sz, train_scores, val_scores = Eval.calc_learning_curves(
                i,  scorer=scorer)

            # plot the learning curve subplot
            out.append(f"{i} plotting {name}. sizes = {train_sz}")
            cls.add_learning_curve_subplot(
                name, i, ax, train_scores, val_scores, train_sz, scaled_y=scaled_y
            )

        print(*out, sep='\n')

        fig.suptitle(title)
        fig.tight_layout()

    @staticmethod
    def add_learning_curve_subplot(name, i, ax1, train_scores, validation_scores, train_sizes, scaled_y=False):
        # get the position of the subplot
        x, y = i//3, i % 3

        # plot the learning curves
        ax1[x, y].plot(train_sizes, train_scores, label='Training')
        ax1[x, y].plot(train_sizes, validation_scores,
                       label='Cross-validation')

        # set the title, x and y labels
        ax1[x, y].set_title(name)
        ax1[x, y].set_xlabel('Training Size')
        ax1[x, y].set_ylabel('Accuracy')
        ax1[x, y].legend(loc='best')

        if scaled_y:
            ax1[x, y].set_ylim([0, 1])

        return ax1

    # --- Confusion Matrix ---
    @classmethod
    def plot_confusion_matrix(cls, Eval, title='Confusion Matrix'):

        inputs = Eval.calc_inputs
        n = len(inputs)
        fig, ax = plt.subplots(n//3, 3)

        for inp in inputs:
            i, name = inp['i'], inp['name']

            # calculate confusion matrix
            cm = Eval.calc_confusion_matrix(i)

            # add the confusion matrix subplot
            cls.add_confusion_matrix_subplot(name, i, ax, cm)

        fig.suptitle(title)
        fig.tight_layout()

    @ classmethod
    def add_confusion_matrix_subplot(cls, name, i, ax2, cm):
        """  cm should be a confusion matrix as Pandas DataFrame """

        # get the position of the subplot
        x, y = i//3, i % 3

        # plot the confusion matrix
        sn.heatmap(cm, annot=True,
                   ax=ax2[x, y],
                   xticklabels=cls.LABELS, yticklabels=cls.LABELS)

        # set the title, x and y labels
        ax2[x, y].set_title(name)
        ax2[x, y].set_xlabel('')
        ax2[x, y].set_ylabel('')

        return ax2

    # --- Classification Report ---
    @classmethod
    def plot_classification_report(cls, Eval, title='Classification Report'):

        inputs = Eval.calc_inputs
        n = len(inputs)

        fig, ax = plt.subplots(n//3, 3, figsize=(8, 8))
        for inp in inputs:
            i, name = inp['i'], inp['name']

            # calculate classification report
            cr = Eval.calc_classification_report(i)

            # add the classification report subplot
            cls.add_classification_report_subplot(name, i, ax, cr)

        fig.suptitle(title)
        fig.tight_layout()

    def add_classification_report_subplot(name, i, ax3, cr):
        """  cr should be a classification report as Pandas DataFrame """

        # get the position of the subplot
        x, y = i//3, i % 3

        # plot the classification report
        sn.heatmap(cr.iloc[-2:-1, :-1], annot=True, ax=ax3[x, y])
        ax3[x, y].set_title(name)
        ax3[x, y].set_xlabel('')
        ax3[x, y].set_ylabel('')

        return ax3
