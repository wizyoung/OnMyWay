import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=2)

# Weizmann
# y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# y_pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# motion = ['Walk', 'Run', 'Jump', 'Side',
#           'Bend', 'Wave1', 'Wave2', 'Pjump',
#           'Jack', 'Skip']

# UCF-sports
# y_true = [0] * 50 + [1] * 60 + [2] * 70 + [3] * 20 + [4] * 40 + \
#          [5] * 40 + [6] * 40 + [7] * 70 + [8] * 40 + [9] * 70
# y_true = np.array(y_true)
# y_pred = y_true.copy()
# y_pred[90:100] = [9] * 10
# y_pred[280:290] = [4] * 10
# y_pred[290:292] = [6] * 2
# y_pred[490:500] = [1] * 10
# print y_true
# motion = ['Diving', 'Golf', 'Kicking', 'Lifting', 'Horsing',
#           'Running', 'Skateboarding', 'Swing-bench', 'Swing-side', 'Walking']

# KTH
# y_true = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100
# y_true = np.array(y_true)
# y_pred = y_true.copy()
# y_pred[0] = 3
# # y_pred[400:406] = [5] * 6
# y_pred[401] = 3
# y_pred[402:408] = [5] * 6
# y_pred[500:503] = [4] * 3
# motion = ['Boxing', 'Handclapping', 'Handwaving', 'Jogging', 'Running', 'Walking']

# VIVA
y_true = [0] * 32 + [1] * 32 + [2] * 32 + [3] * 32 + [4] * 32 + \
         [5] * 32 + [6] * 32 + [7] * 32 + [8] * 32 + [9] * 32 + \
         [10] * 32 + [11] * 32 + [12] * 32 + [13] * 32 + [14] * 32 + \
         [15] * 32 + [16] * 32 + [17] * 32 + [18] * 32
y_true = np.array(y_true)
y_pred = y_true.copy()
y_pred[0:2] = [16] * 2
y_pred[32] = 8
y_pred[33:35] = [15] * 2
y_pred[35:37] = [16] * 2
y_pred[64:66] = [0] * 2
y_pred[66] = 5
y_pred[67] = 18
y_pred[96:98] = [12] * 2
y_pred[128] = 5
y_pred[160:162] = [4] * 2
y_pred[320:322] = [7] * 2
y_pred[384:386] = [7] * 2
y_pred[416:418] = [8] * 2
y_pred[448] = 3
y_pred[544] = 18
motion = ['SwipeR','SwipeL','SwipeD','SwipeU','SwipeV','SwipeX','Swipe+',
          'ScrollR','ScrollL','ScrollD','ScrollU','Tap1','Tap3','Pinch',
          'Expand','RotateCCW','RotateCW','Open','Close']

# --------------OVO-----------------
# KTH2
# y_true = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100
# y_true = np.array(y_true)
# y_pred = y_true.copy()
# y_pred[0:2] = [1] * 2
# y_pred[100:104] = [0] * 4
# y_pred[104:106] = [2] * 2
# y_pred[200:204] = [0] * 4
# y_pred[204:206] = [1] * 2
# y_pred[302:304] = [0] * 2
# y_pred[304:306] = [1] * 2
# y_pred[306:314] = [2] * 8
# y_pred[314:316] = [4] * 2
# y_pred[400:402] = [2] * 2
# y_pred[402:408] = [5] * 6
# motion = ['Boxing', 'Handclapping', 'Handwaving', 'Jogging', 'Running', 'Walking']

# UCF-sports2
# y_true = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
#   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#   1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
#   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
#   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
#   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
#   3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
#   3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
#   4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
#   4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
#   5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
#   5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  4,  6,  6,  6,  6,  6,  6,
#   6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
#   6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,
#   7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
#   7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
#   7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
#   7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,
#   8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
#   8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,
#   9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
#   9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
#   9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
#   9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9]
# y_true = np.array(y_true)
# y_pred = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
#   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#   1,  1,  1,  1,  1,  1,  1,  1,  6,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#   1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
#   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
#   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
#   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
#   3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
#   3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
#   4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
#   4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
#   5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
#   5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  4,  6,  6,  6,  6,  6,  6,
#   6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
#   6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,
#   7,  7,  7,  7,  7,  7,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  7,  7,
#   7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
#   7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
#   7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,
#   8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
#   8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  6,
#   6,  9,  1,  9,  6,  6,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
#   9,  9,  5,  9,  5,  9,  9,  9,  9,  5,  9,  6,  9,  9,  9,  9,  6,  9,
#   9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
#   9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9]
# y_pred = np.array(y_pred)
# motion = ['Diving', 'Golf', 'Kicking', 'Lifting', 'Horsing',
#           'Running', 'Skateboarding', 'Swing-bench', 'Swing-side', 'Walking']


cm = confusion_matrix(y_true, y_pred)
cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
print cm


def plot_cm(cm, classes,
            title='Confusion Matrix', showtitle=False, showcolorbar=False,
            cmap=plt.cm.Greys, fontname='Times New Roman', precision=2):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if showcolorbar:
        plt.colorbar()
    if showtitle:
        plt.title(title, fontname=fontname)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-
               45, fontname=fontname, fontsize=10)
    plt.yticks(tick_marks, classes, fontname=fontname, fontsize=10)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], precision) if cm[i, j] != 0 else 0,
                 horizontalalignment='center', verticalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black', fontsize=7)
    plt.tight_layout()
    # plt.ylabel('True label', fontname=fontname)
    # plt.xlabel('Predicted label', fontname=fontname)


# plt.figure()
plot_cm(cm, motion)
# plt.show()
plt.savefig('/Users/chenyang/Desktop/v.pdf', format='pdf', dpi=300)
plt.close()
