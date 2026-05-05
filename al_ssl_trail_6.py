# Clean reviewer-facing Active Learning + Pseudo-labelling pipeline
# Odd/Even alternation:
#   Even rounds (0, 2, 4...): Active Learning — 50 most uncertain, human labels
#   Odd rounds (1, 3, 5...): Pseudo-labelling — up to 50 confident (>0.90), human verifies & corrects
# Human-in-the-loop: pseudo labels are verified against ground truth, corrected if wrong
# MC Dropout iterations = 15
# Runs for multiple seeds sequentially

import gc
import random
from copy import deepcopy
from datetime import datetime
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torchmetrics import F1Score
from torchvision.models import resnet18
from torchvision.transforms import transforms

from baal.active import ActiveLearningDataset, FileDataset
from baal.active.heuristics import Entropy
from baal.bayesian.dropout import MCDropoutModule
from baal.modelwrapper import ModelWrapper, TrainingArgs
from baal.utils.metrics import Accuracy, ECE, ECE_PerCLs

# ---------------- CONFIG ----------------
SEEDS = [42, 123, 456]
INITIAL_LABELS = 50
AL_QUERY_SIZE = 50
PL_QUERY_SIZE = 50
CONF_THRESHOLD = 0.90
MC_ITERATIONS = 15
TRAIN_EPOCHS = 30
AL_ROUNDS = 30
BATCH_SIZE = 32
LR = 0.001

classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
num_classes = len(classes)

def get_label(img_path):
    return classes.index(img_path.split('/')[-2])

# ============================================================
# RUN FOR EACH SEED
# ============================================================
for SEED in SEEDS:
    print(f'\n\n{"#"*60}')
    print(f'  STARTING EXPERIMENT WITH SEED={SEED}')
    print(f'{"#"*60}\n')

    # ---------------- SEED ----------------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- DATA ----------------
    all_files = glob('/home/achazhoor/Documents/2024/active_learning/data/train/*/*.jpg')
    all_labels = [get_label(f) for f in all_files]
    train, test = train_test_split(all_files, test_size=0.4, random_state=SEED, stratify=all_labels)
    print(f'Train pool: {len(train)}, Test: {len(test)}')

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(200),
        transforms.RandomCrop(200),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(200),
        transforms.RandomCrop(200),
        transforms.ToTensor(),
    ])

    train_dataset = FileDataset(train, [-1] * len(train), train_transform)
    test_dataset = FileDataset(test, [-1] * len(test), test_transform)

    for idx in range(len(test_dataset)):
        test_dataset.label(idx, get_label(test_dataset.files[idx]))

    active_learning_ds = ActiveLearningDataset(
        train_dataset,
        pool_specifics={'transform': test_transform}
    )

    initial_idx = np.random.permutation(np.arange(len(train_dataset)))[:INITIAL_LABELS].tolist()
    initial_labels = [get_label(train_dataset.files[idx]) for idx in initial_idx]
    active_learning_ds.label(initial_idx, initial_labels)
    print(f'Initial labelled: {len(active_learning_ds)} | Unlabelled pool: {active_learning_ds.n_unlabelled}')

    # ---------------- MODEL ----------------
    USE_CUDA = torch.cuda.is_available()
    device = 'cuda' if USE_CUDA else 'cpu'

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = MCDropoutModule(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    args = TrainingArgs(
        criterion=criterion,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        epoch=TRAIN_EPOCHS,
        use_cuda=USE_CUDA,
        replicate_in_memory=False,
    )

    baal_model = ModelWrapper(model=model, args=args)
    baal_model.add_metric('accuracy', lambda: Accuracy())
    baal_model.add_metric('f1', lambda: F1Score(task='multiclass', num_classes=num_classes).to(device))
    baal_model.add_metric('ece', lambda: ECE(n_bins=10))
    baal_model.add_metric('ece_per_cls', lambda: ECE_PerCLs(num_classes))

    init_weights = deepcopy(baal_model.state_dict())
    heuristic = Entropy(shuffle_prop=0)

    pseudo_label_errors = {}

    # ---------------- LOGGING ----------------
    try:
        wandb.init(
            project='AL_SSL_Lap_after_phd',
            name=f'Experiment_seed_{SEED}',
            config={
                'seed': SEED,
                'initial_labels': INITIAL_LABELS,
                'al_query_size': AL_QUERY_SIZE,
                'pl_query_size': PL_QUERY_SIZE,
                'conf_threshold': CONF_THRESHOLD,
                'mc_iterations': MC_ITERATIONS,
                'train_epochs': TRAIN_EPOCHS,
                'al_rounds': AL_ROUNDS,
                'batch_size': BATCH_SIZE,
                'lr': LR,
            },
            reinit=True,
        )
    except Exception:
        print('wandb unavailable, continuing...')

    # ---------------- LOOP ----------------
    for step in range(AL_ROUNDS):
        is_al_round = (step % 2 == 0)
        round_type = "ACTIVE LEARNING" if is_al_round else "PSEUDO LABELING"

        print(f'\n{"="*60}')
        print(f'[Seed={SEED}] AL Round {step+1}/{AL_ROUNDS} [{round_type}]')
        print(f'Training on {len(active_learning_ds)} labelled samples | Pool remaining: {active_learning_ds.n_unlabelled}')
        print(f'{"="*60}')

        baal_model.load_state_dict(init_weights)

        try:
            hist, best_weight = baal_model.train_and_test_on_datasets(
                active_learning_ds,
                test_dataset,
                return_best_weights=True,
                patience=5,
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
            raise

        metrics = baal_model.get_metrics()
        print(f'  Metrics: {metrics}')

        # ---------------- CONFUSION MATRIX + METRICS ----------------
        baal_model.load_state_dict(best_weight)
        test_preds = baal_model.predict_on_dataset(
            test_dataset,
            iterations=1,
            verbose=False,
        )
        test_pred_classes = np.argmax(test_preds.mean(axis=-1), axis=1)
        test_true_labels = np.array([get_label(test_dataset.files[idx]) for idx in range(len(test_dataset))])

        cm = confusion_matrix(test_true_labels, test_pred_classes, labels=list(range(num_classes)))
        per_class_precision = precision_score(test_true_labels, test_pred_classes, average=None, zero_division=0)
        per_class_recall = recall_score(test_true_labels, test_pred_classes, average=None, zero_division=0)
        cls_report = classification_report(test_true_labels, test_pred_classes, target_names=classes, zero_division=0)
        print(cls_report)

        pool = active_learning_ds.pool
        if len(pool) == 0:
            print('Pool exhausted.')
            break

        # ---------------- MC PREDICTION ON POOL ----------------
        torch.cuda.empty_cache()
        gc.collect()

        predictions = baal_model.predict_on_dataset(
            pool,
            iterations=MC_ITERATIONS,
            verbose=False,
        )

        if np.isnan(predictions).any():
            raise ValueError('NaN predictions detected')

        mean_preds = predictions.mean(axis=-1)
        confidence_scores = np.max(mean_preds, axis=1)

        # ---------------- EVEN ROUNDS: ACTIVE LEARNING ----------------
        al_added = 0
        pseudo_added = 0

        if is_al_round:
            top_uncertainty = heuristic(predictions)[:AL_QUERY_SIZE]
            oracle_indices = active_learning_ds._pool_to_oracle_index(top_uncertainty.tolist())
            al_labels = [get_label(train_dataset.files[idx]) for idx in oracle_indices]
            active_learning_ds.label(top_uncertainty.tolist(), al_labels)
            al_added = len(top_uncertainty)

            print(f'  [AL] Selected {al_added} most uncertain samples')
            print(f'  [AL] Human labelled and added to training pool')

        # ---------------- ODD ROUNDS: PSEUDO LABELING (Human verified) ----------------
        else:
            confident_idx = np.where(confidence_scores > CONF_THRESHOLD)[0]

            if len(confident_idx) > 0:
                certainty_scores = confidence_scores[confident_idx]
                top_idx = np.argsort(certainty_scores)[-PL_QUERY_SIZE:]
                chosen_pool_indices = [int(confident_idx[i]) for i in top_idx]

                oracle_indices = active_learning_ds._pool_to_oracle_index(chosen_pool_indices)

                pseudo_labels = np.argmax(mean_preds[chosen_pool_indices], axis=1)

                correct_labels = [get_label(train_dataset.files[idx]) for idx in oracle_indices]
                num_incorrect = sum(
                    1 for pl, gt in zip(pseudo_labels, correct_labels) if pl != gt
                )
                pseudo_label_errors[step + 1] = num_incorrect

                print(f'  [PL] Confident candidates (>{CONF_THRESHOLD}): {len(confident_idx)}')
                print(f'  [PL] Selected top {len(chosen_pool_indices)} for human verification')
                print(f'  [PL] Pseudo labels incorrect: {num_incorrect}/{len(chosen_pool_indices)}')
                print(f'  [PL] Human corrects all labels -> adding GROUND TRUTH labels')

                active_learning_ds.label(chosen_pool_indices, correct_labels)
                pseudo_added = len(chosen_pool_indices)
            else:
                print(f'  [PL] No samples above confidence threshold {CONF_THRESHOLD}')
                pseudo_label_errors[step + 1] = 0

        print(f'  [TOTAL] Labelled pool now: {len(active_learning_ds)} | Remaining unlabelled: {active_learning_ds.n_unlabelled}')

        # ---------------- WANDB LOGGING ----------------
        try:
            log_dict = {
                'round': step + 1,
                'round_type': round_type,
                'labelled_pool': len(active_learning_ds),
                'al_added': al_added,
                'pseudo_added': pseudo_added,
                'pseudo_incorrect': pseudo_label_errors.get(step + 1, 0),
                'test_accuracy': metrics.get('test_accuracy', None),
                'test_f1': metrics.get('test_f1', None),
                'test_ece': metrics.get('test_ece', None),
                'test_loss': metrics.get('test_loss', None),
                'train_loss': metrics.get('train_loss', None),
                'confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=test_true_labels.tolist(),
                    preds=test_pred_classes.tolist(),
                    class_names=classes,
                ),
            }
            for i, cls_name in enumerate(classes):
                log_dict[f'precision/{cls_name}'] = per_class_precision[i]
                log_dict[f'recall/{cls_name}'] = per_class_recall[i]

            wandb.log(log_dict)
        except Exception:
            pass

        baal_model._reset_metrics()

    # ---------------- SUMMARY FOR THIS SEED ----------------
    print(f'\n{"="*60}')
    print(f'PSEUDO LABEL ERROR SUMMARY (Seed={SEED}):')
    print(f'{"="*60}')
    for round_num, errors in sorted(pseudo_label_errors.items()):
        print(f'  Round {round_num}: {errors} incorrect pseudo labels corrected by human')
    print(f'  Total corrections: {sum(pseudo_label_errors.values())}')

    try:
        wandb.finish()
    except Exception:
        pass

print('\n\nAll experiments finished.')