import os
import h5py
import pickle
from argparse import ArgumentParser
import numpy as np

import torch
from torch.distributions import kl_divergence, Normal

import constants
from AdversarialVAE import Classifier, AdversarialVAE
from arguments import generate_cfg_adversarial
from data_converter import RollAugMonoSingleSequenceConverter
from note_sequence_ops import steps_per_quarter_to_seconds_per_step, quantize_note_sequence
from midi_io import note_sequence_to_pretty_midi
from chords_infer import infer_chords_for_sequence
from note_sequence import Tempo, TimeSignature
from utils import std_normal

# CHANGE PATH HERE
output_path_base = 'output'


def to_tensor(data, device):
    if device == 'cpu':
        return torch.LongTensor(data, device=device)
    return torch.cuda.LongTensor(data, device=device)


def to_float_tensor(data, device):
    if device == 'cpu':
        return torch.FloatTensor(data, device=device)
    return torch.cuda.FloatTensor(data, device=device)


def create_category_const(attrs, batch_size, thresholds):
    if len(attrs) == 1:
        category_ratios = [[thre] for thre in thresholds[attrs[0]]]
        names = ['_{}{:.2}'.format(attrs[0], ratio[0]) for ratio in category_ratios]
        padding = [[0.] for _ in range(batch_size - len(category_ratios) - 1)]
        category_const = np.array(category_ratios + padding)
        category_const_cls = np.array([[i] for i in range(len(thresholds[attrs[0]]))] + padding)
    else:
        raise Exception('unsupported number of category')
    return category_const, category_const_cls, names


def get_original_category_name(attrs, original_category):
    if len(attrs) == 1:
        return '_org_{}{:.2}'.format(attrs[0], float(original_category[0][0]))
    if len(attrs) == 2:
        return '_org_{}{:.2}_{}{:.2}'.format(attrs[0], float(original_category[0][0]),
                                             attrs[1], float(original_category[0][1]))
    if len(attrs) == 3:
        return '_org_{}{:.2}_{}{:.2}_{}{:.2}'.format(attrs[0], float(original_category[0][0]),
                                                     attrs[1], float(original_category[0][1]),
                                                     attrs[2], float(original_category[0][2]))

def test(a_chord_test, a_category_test, a_category_cls_test, a_event_test, keys_test, model, model_d, device, attrs,
         batch_size, thresholds):
    model.eval()
    model_d.eval()
    losses, losses_d, losses_kl, accs = 0., 0., 0., 0.
    preds = {}

    category_const, category_const_cls, names = create_category_const(attrs, batch_size, thresholds)
    with torch.no_grad():
        for key in keys_test:
            chord_tensor = to_float_tensor(a_chord_test[key], device=device).repeat(batch_size, 1, 1)
            event_tensor = to_tensor(a_event_test[key], device=device).repeat(batch_size, 1, 1)

            original_category_value = np.array([a_category_test[attr + '/' + key] for attr in attrs]).reshape(1, -1)
            category_tensor = np.concatenate([original_category_value, category_const])
            category_tensor = to_float_tensor(category_tensor, device=device)

            original_category_cls_value = np.array([a_category_cls_test[attr + '/' + key] for attr in attrs]).reshape(1, -1)
            category_cls_tensor = np.concatenate([original_category_cls_value, category_const_cls])
            category_cls_tensor = to_tensor(category_cls_tensor, device=device)

            loss, pred, lv, acc, distribution = model(event_tensor, chord_tensor, category_tensor)
            dis_out = model_d(lv)
            loss_d = model_d.calc_loss(dis_out, category_cls_tensor)

            normal = std_normal(distribution.mean.size())
            loss_kl = kl_divergence(distribution, normal).mean()

            losses += loss
            accs += acc
            losses_d += loss_d
            losses_kl += loss_kl

            original_category_name = get_original_category_name(attrs, original_category_value)
            preds[key + original_category_name] = pred[0]
            for i, name in enumerate(names):
                preds[key + name] = pred[i + 1]

    return losses.item() / batch_size, losses_d.item() / batch_size, losses_kl.item() / batch_size, accs / batch_size,\
           preds


def evaluation(model, model_d, cfg, device):
    a_chord_test = h5py.File(cfg['data']['chord_f_valid'], 'r')
    a_event_test = h5py.File(cfg['data']['event_valid'], 'r')
    a_category_test = h5py.File(cfg['data']['attr_valid'], 'r')
    a_category_cls_test = h5py.File(cfg['data']['attr_cls_valid'], 'r')
    with open(cfg['data']['keys_valid'], 'rb') as f:
        keys_test = pickle.load(f)

    threshold_path = '/data/.../category_cls_thresholds.pkl'
    with open(threshold_path, 'rb') as f:
        thresholds = pickle.load(f, encoding='latin1')
    return test(a_chord_test, a_category_test, a_category_cls_test, a_event_test, keys_test, model, model_d, device,
                cfg['attr'], cfg['batch_size'], thresholds)


def run(args):
    output_dir = os.path.join(output_path_base, args.model_name)
    latest_model_text_file = os.path.join(output_dir, 'latest_model.txt')
    sample_dir = os.path.join(output_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    if not os.path.exists(output_dir) and not os.path.exists(latest_model_text_file):
        raise IOError("Model file not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.current_device())

    if args.model_path:
        latest_model_path = args.model_path
    else:
        with open(latest_model_text_file, 'r') as f:
            latest_model_path = f.read()
    checkpoint = torch.load(latest_model_path)
    latest_model_name = latest_model_path.split('/')[-1]
    cfg = generate_cfg_adversarial(None, args, output_dir, checkpoint)

    print(cfg)

    model_d = Classifier(input_dim=cfg['z_dims'],
                         num_layers=cfg['model']['discriminator']['num_layers'],
                         n_attr=len(cfg['attr']),
                         activation=cfg['activation_d'],
                         n_classes=8,
                         device=device)

    model = AdversarialVAE(vocab_size=cfg['vocab_size'],
                     hidden_dims=cfg['model']['encoder']['hidden_size'],
                     z_dims=cfg['z_dims'],
                     n_step=cfg['bars_per_data'] * cfg['steps_per_quarter'] * constants.QUARTERS_PER_BAR,
                     device=device,
                     n_attr=cfg['n_attr'])

    model.load_state_dict(checkpoint['model'])
    model_d.load_state_dict(checkpoint['model_d'])
    model.to(device)
    model_d.to(device)
    print(model)

    losses, losses_d, losses_kl, accs, preds = evaluation(model, model_d, cfg, device)

    val_result_path = os.path.join(output_dir, 'eval_result.txt')
    with open(val_result_path, 'a') as f:
        f.write("model: {}, val_loss: {}, val_loss_d: {}, val_loss_kl: {}, acc: {}, ".format(
            latest_model_name, losses, losses_d, losses_kl, accs))

    RPSC = RollAugMonoSingleSequenceConverter(steps_per_quarter=cfg['steps_per_quarter'],
                                              quarters_per_bar=constants.QUARTERS_PER_BAR,
                                              chords_per_bar=cfg['chords_per_bar'],
                                              bars_per_data=cfg['bars_per_data'])

    sample_save_path = os.path.join(sample_dir, latest_model_name + '_inter')
    if not os.path.exists(sample_save_path):
        os.mkdir(sample_save_path)

    seconds_per_step = steps_per_quarter_to_seconds_per_step(cfg['steps_per_quarter'], 60)

    f_chord_test = h5py.File(cfg['data']['chord_valid'], 'r')
    f_event_test = h5py.File(cfg['data']['event_valid'], 'r')
    with open(cfg['data']['keys_valid'], 'rb') as f:
        keys_test = pickle.load(f)

    chord_acc, chord_category_acc = 0., 0.
    for key, event in preds.items():
        # Normalized
        event = event.reshape(cfg['chords_per_data'], -1)
        ns = RPSC.to_note_sequence_from_events(event, seconds_per_step)  # Normalized
        tempo = Tempo(time=0., qpm=60)
        ns.tempos.append(tempo)
        time_signature = TimeSignature(time=0, numerator=4, denominator=4)
        ns.time_signatures.append(time_signature)

        quantized_ns = quantize_note_sequence(ns, steps_per_quarter=cfg['steps_per_quarter'])
        try:
            ns_with_chord = infer_chords_for_sequence(quantized_ns, chords_per_bar=cfg['chords_per_bar'])
        except Exception as e:
            print(e)
            continue
        pm = note_sequence_to_pretty_midi(ns_with_chord)
        key_string = '_'.join(key.split('/'))
        output_path = os.path.join(sample_save_path, key_string + '.mid')

        print(output_path)
        pm.write(output_path)

        chord_list = [ta_chord.text for ta_chord in ns_with_chord.text_annotations]
        chord_txt = ",".join(chord_list)
        output_chord_path = os.path.join(sample_save_path, key_string + '.txt')
        with open(output_chord_path, 'w') as f:
            f.write(chord_txt)

    with open(val_result_path, 'a') as f:
        f.write("chord_acc: {}\n".format(chord_acc / len(keys_test)))

    original_path = os.path.join(sample_dir, 'original')
    if not os.path.exists(original_path):
        os.mkdir(original_path)

        for key in keys_test:
            # Unnormalized
            ns = RPSC.to_note_sequence_from_events(np.array(f_event_test[key]), seconds_per_step)
            pm = note_sequence_to_pretty_midi(ns)
            key_string = '_'.join(key.split('/'))
            output_path = os.path.join(original_path, key_string + '.mid')
            pm.write(output_path)
            chord = list(f_chord_test[key])
            chord_list = [RPSC.chord_from_index(c) for c in chord]
            chord_list = ",".join(chord_list)
            output_chord_path = os.path.join(original_path, key_string + '.txt')
            with open(output_chord_path, 'w') as f:
                f.write(chord_list)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=[0, 1], nargs='+', help='used gpu')
    parser.add_argument('--model_name', type=str, default="tmp", help='model name')
    parser.add_argument('--model_path', type=str, help='to use a specific model, not latest')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)

    run(args)
