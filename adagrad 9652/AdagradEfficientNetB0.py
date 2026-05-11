#Trained by Faiz Behzad 9652
import os
import sys
import glob
import time
import platform
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

# ── Speed optimizations (must be set before TF import) ───────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "2"
os.environ["TF_GPU_THREAD_MODE"]     = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"]    = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

import tensorflow as tf

# ── GPU setup ────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅  GPU detected: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"GPU config error: {e}")
else:
    print("⚠️  No GPU detected – training will use CPU (slow).")

# ── Mixed Precision (biggest speedup for RTX 4060 Tensor Cores) ──────────────
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✅  Mixed precision policy:", mixed_precision.global_policy().name)

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adagrad
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score,
                             classification_report)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

print("=" * 80)
print("Real Field TOMATO Disease Classification – Local RTX 4060 Training (EfficientNetB0)")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = r"H:\STICS\TomatoTrain"
DATASET_PATH = os.path.join(BASE_DIR, "Real_Field_Tomato_Dataset")

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 128
RANDOM_STATE = 44322

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR   = os.path.join(DATASET_PATH, "validation")
TEST_DIR  = os.path.join(DATASET_PATH, "test")

LR_PHASE1 = 0.01
LR_PHASE2 = 0.001

seed = RANDOM_STATE
np.random.seed(seed)
tf.random.set_seed(seed)

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
for p in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.isdir(p):
        sys.exit(f"❌  Directory not found: {p}")

print(f"\nDataset root : {DATASET_PATH}")
print(f"Train        : {TRAIN_DIR}")
print(f"Validation   : {VAL_DIR}")
print(f"Test         : {TEST_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def compute_weights_from_dir(train_dir):
    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    print(f"\nClasses found ({len(class_names)}): {class_names}")

    counts = {}
    extensions = ('*.jpg','*.jpeg','*.png','*.bmp',
                  '*.JPG','*.JPEG','*.PNG','*.BMP')
    for cls in class_names:
        files = []
        for ext in extensions:
            files += glob.glob(os.path.join(train_dir, cls, ext))
        counts[cls] = len(files)
        print(f"  {cls}: {len(files)} images")

    if any(v == 0 for v in counts.values()):
        sys.exit("❌  One or more classes have 0 images in the train folder!")

    max_c, min_c = max(counts.values()), min(counts.values())
    ratio = max_c / min_c
    print(f"\nImbalance ratio: {ratio:.2f}:1  "
          f"({'imbalanced' if ratio > 2 else 'balanced'})")

    y = []
    for i, cls in enumerate(class_names):
        y.extend([i] * counts[cls])
    y = np.array(y)

    weights_arr = compute_class_weight('balanced',
                                       classes=np.arange(len(class_names)),
                                       y=y)
    class_weights = {i: float(w) for i, w in enumerate(weights_arr)}
    print("\nClass weights:")
    for i, (cls, w) in enumerate(zip(class_names, weights_arr)):
        print(f"  [{i}] {cls}: {w:.4f}")

    return class_names, class_weights, counts, ratio

class_names, class_weights, train_counts, imbalance_ratio = \
    compute_weights_from_dir(TRAIN_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# tf.data PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(directory, augment=False, shuffle=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=seed,
        label_mode='categorical'
    )

    ds_class_names = ds.class_names

    @tf.function
    def preprocess(images, labels):
        images = preprocess_input(images)
        return images, labels

    @tf.function
    def augment_fn(images, labels):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, max_delta=0.2)
        images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
        images = tf.image.random_crop(
            tf.image.resize_with_crop_or_pad(images, IMG_SIZE[0]+20, IMG_SIZE[1]+20),
            size=(tf.shape(images)[0], IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        return images, labels

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)

    return ds, ds_class_names

print("\nBuilding tf.data pipelines …")
train_ds, _train_classes = make_dataset(TRAIN_DIR, augment=True,  shuffle=True)
val_ds,   _val_classes   = make_dataset(VAL_DIR,   augment=False, shuffle=False)
test_ds,  _test_classes  = make_dataset(TEST_DIR,  augment=False, shuffle=False)

def count_samples(ds):
    total = 0
    for _, y in ds:
        total += y.shape[0]
    return total

print("Counting samples (one pass) …")
train_samples = count_samples(train_ds)
val_samples   = count_samples(val_ds)
test_samples  = count_samples(test_ds)
num_classes   = len(_train_classes)
steps_per_epoch = train_samples // BATCH_SIZE
ordered_classes = sorted(_train_classes)

print(f"\nPipelines ready | classes={num_classes} | "
      f"train={train_samples} | val={val_samples} | test={test_samples}")
print(f"Batch size: {BATCH_SIZE} | Steps/epoch: {steps_per_epoch}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding EfficientNetB0 model …")
base_model = EfficientNetB0(weights='imagenet', include_top=False,
                            input_shape=(*IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation='softmax', dtype='float32')(x)
model = Model(inputs=base_model.input, outputs=out)

for layer in base_model.layers:
    layer.trainable = False

print(f"Total parameters : {model.count_params():,}")

# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS & COMPILE
# ══════════════════════════════════════════════════════════════════════════════
ckpt_path = os.path.join(BASE_DIR, "best_model.weights.h5")

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_best_only=True,
    monitor='val_accuracy',
    save_weights_only=True,
    verbose=1)

model.compile(
    optimizer=Adagrad(learning_rate=LR_PHASE1),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 – frozen base
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("PHASE 1 – Training head (base frozen) for up to 20 epochs")
print("─"*60)
start_time = time.time()

h1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights,
    verbose=1)

p1_train_acc = h1.history['accuracy'][-1]
p1_val_acc   = h1.history['val_accuracy'][-1]
p1_epochs    = len(h1.history['accuracy'])
print(f"\nPhase 1 done | epochs={p1_epochs} | "
      f"train_acc={p1_train_acc:.4f} | val_acc={p1_val_acc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 – fine-tuning (conditional)
# ══════════════════════════════════════════════════════════════════════════════
phase2_ran = False
p2_train_acc = p2_val_acc = None
p2_epochs = 0

if p1_train_acc >= 0.80:
    print("\n" + "─"*60)
    print("PHASE 2 – Fine-tuning top layers of EfficientNetB0 (up to 30 epochs)")
    print("─"*60)
    phase2_ran = True

    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=Adagrad(learning_rate=LR_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    early_stop2 = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True)

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[early_stop2, reduce_lr, checkpoint],
        class_weight=class_weights,
        verbose=1)

    p2_train_acc = h2.history['accuracy'][-1]
    p2_val_acc   = h2.history['val_accuracy'][-1]
    p2_epochs    = len(h2.history['accuracy'])
    print(f"\nPhase 2 done | epochs={p2_epochs} | "
          f"train_acc={p2_train_acc:.4f} | val_acc={p2_val_acc:.4f}")

    combined_acc      = h1.history['accuracy']     + h2.history['accuracy']
    combined_val_acc  = h1.history['val_accuracy'] + h2.history['val_accuracy']
    combined_loss     = h1.history['loss']         + h2.history['loss']
    combined_val_loss = h1.history['val_loss']     + h2.history['val_loss']
else:
    print(f"\nPhase 1 accuracy {p1_train_acc:.2%} < 80% – skipping fine-tuning.")
    combined_acc      = h1.history['accuracy']
    combined_val_acc  = h1.history['val_accuracy']
    combined_loss     = h1.history['loss']
    combined_val_loss = h1.history['val_loss']

train_time_sec = time.time() - start_time
total_epochs   = p1_epochs + p2_epochs
print(f"\nTotal training time: {train_time_sec/60:.1f} minutes | "
      f"total epochs: {total_epochs}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODEL  (weights-only to avoid EagerTensor JSON serialization crash)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving model …")

# Strip any EagerTensors from optimizer variables before saving
try:
    for var in model.optimizer.variables():
        val = var.numpy()
        var.assign(tf.constant(val, dtype=var.dtype))
    current_lr = float(model.optimizer.lr.numpy())
    model.optimizer.lr.assign(current_lr)
    print(f"  Optimizer state sanitized (lr={current_lr:.2e})")
except Exception as e:
    print(f"  Optimizer sanitize warning (non-fatal): {e}")

weights_path = os.path.join(BASE_DIR, "tomato_efficientnetb0_final.weights.h5")
model.save_weights(weights_path)
print(f"  Weights saved → {weights_path}")

# Save metadata (class names + architecture params) for later inference
import json as _json
meta = {
    'class_names' : ordered_classes,
    'num_classes' : num_classes,
    'img_size'    : list(IMG_SIZE),
}
meta_path = os.path.join(BASE_DIR, "model_meta.json")
with open(meta_path, 'w') as _f:
    _json.dump(meta, _f, indent=2)
print(f"  Metadata saved → {meta_path}")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\nEvaluating on test set …")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

# Single-image inference timing
timing_times = []
for images, _ in test_ds.take(10):
    single = images[0:1]
    t0 = time.time()
    model.predict(single, verbose=0)
    timing_times.append(time.time() - t0)
avg_inf_ms = np.mean(timing_times) * 1000 if timing_times else 0.0
print(f"Avg single-image inference: {avg_inf_ms:.2f} ms")

# Full predictions
print("Collecting predictions …")
all_preds, all_labels = [], []
for images, labels in test_ds:
    all_preds.append(model.predict(images, verbose=0))
    all_labels.append(labels.numpy())

preds       = np.concatenate(all_preds,  axis=0)
true_oh     = np.concatenate(all_labels, axis=0)
pred_cls    = np.argmax(preds,   axis=1)
true_labels = np.argmax(true_oh, axis=1)
n = len(pred_cls)

precision = precision_score(true_labels, pred_cls, average='weighted', zero_division=0)
recall    = recall_score   (true_labels, pred_cls, average='weighted', zero_division=0)
f1        = f1_score       (true_labels, pred_cls, average='weighted', zero_division=0)
cm        = confusion_matrix(true_labels, pred_cls)

try:
    auc = roc_auc_score(true_oh, preds, multi_class='ovr')
except Exception:
    auc = float('nan')

print(f"\n{'─'*50}")
print("PER-CLASS METRICS")
print(f"{'─'*50}")
prec_pc = precision_score(true_labels, pred_cls, average=None, zero_division=0)
rec_pc  = recall_score   (true_labels, pred_cls, average=None, zero_division=0)
f1_pc   = f1_score       (true_labels, pred_cls, average=None, zero_division=0)
print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("─"*55)
for i, cls in enumerate(ordered_classes):
    if i < len(prec_pc):
        print(f"{cls:<25} {prec_pc[i]:>10.4f} {rec_pc[i]:>10.4f} {f1_pc[i]:>10.4f}")
print(f"\n{classification_report(true_labels, pred_cls, target_names=ordered_classes, zero_division=0)}")

rng = np.random.default_rng(seed)
boot_accs = [
    np.mean(pred_cls[rng.integers(0, n, n)] == true_labels[rng.integers(0, n, n)])
    for _ in range(1000)
]
ci        = np.percentile(boot_accs, [2.5, 97.5])
boot_std  = float(np.std(boot_accs))
boot_mean = float(np.mean(boot_accs))

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ordered_classes, yticklabels=ordered_classes)
plt.title('Confusion Matrix – Real Field Tomato Disease', fontsize=14, fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"Confusion matrix saved → {cm_path}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(combined_acc,     label='Train')
plt.plot(combined_val_acc, label='Val')
plt.axvline(p1_epochs - 1, color='gray', linestyle='--', alpha=0.5, label='Phase 2 start')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(combined_loss,     label='Train')
plt.plot(combined_val_loss, label='Val')
plt.axvline(p1_epochs - 1, color='gray', linestyle='--', alpha=0.5)
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
hist_path = os.path.join(BASE_DIR, "training_history.png")
plt.savefig(hist_path, dpi=150)
plt.show()
print(f"Training history saved → {hist_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CSV RESULTS
# ══════════════════════════════════════════════════════════════════════════════
mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

results = {
    'timestamp'              : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'seed'                   : seed,
    'os'                     : platform.platform(),
    'tensorflow_version'     : tf.__version__,
    'gpu'                    : str(gpus),
    'mixed_precision'        : 'mixed_float16',
    'base_model'             : 'EfficientNetB0',
    'optimizer'              : 'Adagrad',
    'lr_phase1'              : LR_PHASE1,
    'lr_phase2'              : LR_PHASE2,
    'batch_size'             : BATCH_SIZE,
    'image_size'             : f'{IMG_SIZE[0]}x{IMG_SIZE[1]}',
    'num_classes'            : num_classes,
    'train_samples'          : train_samples,
    'val_samples'            : val_samples,
    'test_samples'           : test_samples,
    'imbalance_ratio'        : round(imbalance_ratio, 4),
    'class_weights'          : str(class_weights),
    'total_epochs'           : total_epochs,
    'phase1_epochs'          : p1_epochs,
    'phase1_final_train_acc' : round(p1_train_acc, 4),
    'phase1_final_val_acc'   : round(p1_val_acc, 4),
    'phase2_ran'             : phase2_ran,
    'phase2_epochs'          : p2_epochs,
    'phase2_final_train_acc' : round(p2_train_acc, 4) if p2_train_acc else None,
    'phase2_final_val_acc'   : round(p2_val_acc,   4) if p2_val_acc   else None,
    'test_accuracy'          : round(float(test_accuracy), 4),
    'test_loss'              : round(float(test_loss), 4),
    'precision_weighted'     : round(float(precision), 4),
    'recall_weighted'        : round(float(recall), 4),
    'f1_weighted'            : round(float(f1), 4),
    'auc_ovr'                : round(float(auc), 4) if not np.isnan(auc) else None,
    'bootstrap_mean_acc'     : round(boot_mean, 4),
    'bootstrap_std_acc'      : round(boot_std, 4),
    'ci_lower'               : round(float(ci[0]), 4),
    'ci_upper'               : round(float(ci[1]), 4),
    'avg_inference_ms'       : round(avg_inf_ms, 3),
    'training_time_minutes'  : round(train_time_sec / 60, 2),
    'memory_used_mb'         : round(mem_mb, 2),
    'data_pipeline'          : 'tf.data + cache + prefetch',
    'model_saved'            : weights_path,
}

csv_path = os.path.join(BASE_DIR, "results.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results.keys())
    w.writeheader()
    w.writerow(results)
print(f"Results saved → {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)
print(f"Test Accuracy  : {test_accuracy:.4f}  (±{boot_std:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}])")
print(f"Precision (w)  : {precision:.4f}")
print(f"Recall    (w)  : {recall:.4f}")
print(f"F1        (w)  : {f1:.4f}")
print(f"AUC (OvR)      : {auc:.4f}" if not np.isnan(auc) else "AUC (OvR)      : N/A")
print(f"Inf time       : {avg_inf_ms:.2f} ms / image")
print(f"Train time     : {train_time_sec/60:.1f} min")
print(f"Weights saved  : {weights_path}")
print(f"Metadata saved : {meta_path}")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════
def predict_tomato_disease(image_path, loaded_model=None):
    m = loaded_model or model
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = preprocess_input(np.expand_dims(arr, 0))
    probs = m.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return ordered_classes[idx], float(probs[idx]), \
           {ordered_classes[i]: float(probs[i]) for i in range(len(ordered_classes))}

print("\nExample usage after training:")
print("  cls, conf, all_p = predict_tomato_disease(r'path\\to\\leaf.jpg')")
print("  print(f'Predicted: {cls}  ({conf:.2%} confidence)')")

# ══════════════════════════════════════════════════════════════════════════════
# HOW TO RELOAD THIS MODEL LATER (printed as a reminder)
# ══════════════════════════════════════════════════════════════════════════════
print("""
─────────────────────────────────────────────────────
TO RELOAD THE MODEL IN A NEW SCRIPT:
─────────────────────────────────────────────────────
import json, numpy as np, tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

BASE_DIR     = r"H:\\STICS\\TomatoTrain"
weights_path = BASE_DIR + r"\\tomato_efficientnetb0_final.weights.h5"
meta_path    = BASE_DIR + r"\\model_meta.json"

with open(meta_path) as f:
    meta = json.load(f)
ordered_classes = meta['class_names']
num_classes     = meta['num_classes']
IMG_SIZE        = tuple(meta['img_size'])

base_model = EfficientNetB0(weights='imagenet', include_top=False,
                            input_shape=(*IMG_SIZE, 3))
x   = base_model.output
x   = GlobalAveragePooling2D()(x)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.5)(x)
x   = Dense(128, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(num_classes, activation='softmax', dtype='float32')(x)
model = Model(inputs=base_model.input, outputs=out)
model.load_weights(weights_path)
print("Model loaded ✅")
─────────────────────────────────────────────────────
""")