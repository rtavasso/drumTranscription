import os
from datetime import datetime

import collections
import pprint as _pprint

try:
    from collections import abc as _collections_abc
except ImportError:  # pragma: no cover - very old Pythons only
    _collections_abc = None

if _collections_abc is not None:
    for _alias in ("Mapping", "MutableMapping", "MutableSequence", "Sequence", "Iterable"):
        if hasattr(collections, _alias):
            continue
        if hasattr(_collections_abc, _alias):
            setattr(collections, _alias, getattr(_collections_abc, _alias))

if not hasattr(_pprint, "_safe_repr"):
    def _safe_repr(obj, context, maxlevels, level):
        """Fallback that mimics the stdlib pprint._safe_repr contract."""

        try:
            return repr(obj), True, False
        except Exception:  # pragma: no cover - best-effort fallback
            return object.__repr__(obj), False, False

    _pprint._safe_repr = _safe_repr

import numpy as np
import wandb
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *

ex = Experiment('train_transcriber')


@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'MAESTRO'
    dataset_path = None

    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, dataset_path):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    def _init_wandb_run():
        project = os.environ.get('WANDB_PROJECT', 'onsets-and-frames')
        wandb_kwargs = dict(
            project=project,
            name=os.path.basename(os.path.normpath(logdir)),
            dir=logdir,
            reinit=True,
        )
        entity = os.environ.get('WANDB_ENTITY')
        if entity:
            wandb_kwargs['entity'] = entity

        try:
            run = wandb.init(**wandb_kwargs)
        except Exception as exc:  # pragma: no cover - network/auth issues
            print(f'Warning: Failed to initialize Weights & Biases logging ({exc}). Continuing without it.')
            return None

        try:
            run.config.update(dict(ex.current_run.config), allow_val_change=True)
        except Exception:
            pass
        return run

    wandb_run = _init_wandb_run()

    def _scalarize(value):
        if hasattr(value, 'item'):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, (np.ndarray, np.generic)):
            value = float(np.asarray(value))
        return value

    def _log_to_wandb(metrics, step):
        nonlocal wandb_run
        if wandb_run is None or not metrics:
            return
        try:
            wandb_run.log(metrics, step=step)
        except Exception as exc:
            print(f'Warning: Failed to log metrics to Weights & Biases ({exc}). Disabling logging.')
            wandb_run = None

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        if train_on != 'MAESTRO':
            raise ValueError('leave_one_out is only supported for the MAESTRO dataset')
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    dataset_name = train_on.upper()
    dataset_kwargs = dict(sequence_length=sequence_length)
    validation_kwargs = dict(sequence_length=validation_length)
    if dataset_path is not None:
        dataset_kwargs['path'] = dataset_path
        validation_kwargs['path'] = dataset_path

    if dataset_name == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, **dataset_kwargs)
        validation_dataset = MAESTRO(groups=validation_groups, **validation_kwargs)
    elif dataset_name == 'MAPS':
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], **dataset_kwargs)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], **validation_kwargs)
    elif dataset_name in {'SLAKH', 'BABYSLAKH', 'BABYSLAKH_DRUMS'}:
        dataset = BabySlakhDrums(groups=train_groups, **dataset_kwargs)
        validation_dataset = BabySlakhDrums(groups=validation_groups, **validation_kwargs)
    else:
        raise ValueError(f'Unsupported dataset: {train_on}')

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        train_metrics = {}
        for key, value in {'loss': loss, **losses}.items():
            scalar_value = _scalarize(value)
            writer.add_scalar(key, scalar_value, global_step=i)
            train_metrics[key] = scalar_value

        if hasattr(scheduler, 'get_last_lr'):
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, global_step=i)
        train_metrics['learning_rate'] = current_lr

        _log_to_wandb(train_metrics, i)

        if i % validation_interval == 0:
            model.eval()
            val_metrics = {}
            with torch.no_grad():
                for key, value in evaluate(validation_dataset, model).items():
                    metric_name = 'validation/' + key.replace(' ', '_')
                    metric_value = float(np.mean(value))
                    writer.add_scalar(metric_name, metric_value, global_step=i)
                    val_metrics[metric_name] = metric_value
            _log_to_wandb(val_metrics, i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
