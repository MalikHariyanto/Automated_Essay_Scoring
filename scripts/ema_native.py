"""Visible E53 EMA recurrence, resume state, and serialized report checks."""


def validate_ema_config(cfg):
    if cfg['inference_weights'] != 'ema' or cfg['ema_update_policy'] != 'successful_active_optimizer_step':
        raise ValueError('E53 memakai bobot EMA dan update hanya setelah optimizer berhasil dengan LR aktif.')
    if isinstance(cfg['ema_decay_max'], bool) or not math.isfinite(cfg['ema_decay_max']) or cfg['ema_decay_max'] != 0.99:
        raise ValueError('E53 mengunci batas decay EMA 0,99.')
    if cfg['ema_warmup_offset'] != 10 or isinstance(cfg['ema_warmup_offset'], bool):
        raise ValueError('E53 mengunci offset warmup EMA 10.')


def ema_decay_at(update, maximum=0.99, offset=10):
    """First successful update copies online weights; subsequent updates use ramped EMA."""
    if type(update) is not int or update < 1:
        raise ValueError('Nomor update EMA harus integer positif.')
    return 0.0 if update == 1 else min(maximum, (1.0+update)/(offset+update))


class ExponentialWeightAverage:
    """One detached FP32 evaluation copy; online model keeps receiving MSE gradients."""
    def __init__(self, model, cfg):
        validate_ema_config(cfg)
        self.maximum = float(cfg['ema_decay_max'])
        self.offset = int(cfg['ema_warmup_offset'])
        # deepcopy preserves shared initialization and does not consume random numbers.
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update_after_optimizer(self, online_model, *, lr_active, skipped):
        # AMP overflow or zero LR must not advance the EMA clock.
        if skipped or not lr_active:
            return False
        online = dict(online_model.named_parameters())
        shadow = dict(self.model.named_parameters())
        if set(online) != set(shadow):
            raise ValueError('Nama parameter online/EMA tidak cocok.')
        self.updates += 1
        decay = ema_decay_at(self.updates, self.maximum, self.offset)
        for name, averaged in shadow.items():
            current = online[name].detach()
            if current.shape != averaged.shape or current.dtype != averaged.dtype:
                raise ValueError('Shape/dtype parameter online/EMA berbeda.')
            if self.updates == 1:
                averaged.copy_(current)
            else:
                averaged.lerp_(current, 1.0-decay)
        # Position/token-type buffers are copied, never averaged as integer tensors.
        buffers = dict(online_model.named_buffers())
        for name, averaged in self.model.named_buffers():
            averaged.copy_(buffers[name])
        self.model.eval()
        return True

    def state_dict(self):
        return {'format': 'ema_state_v1', 'maximum': self.maximum, 'offset': self.offset,
                'updates': self.updates, 'model': self.model.state_dict()}

    def load_state_dict(self, payload):
        if (payload.get('format') != 'ema_state_v1' or payload.get('maximum') != self.maximum
                or payload.get('offset') != self.offset or type(payload.get('updates')) is not int
                or payload['updates'] < 0):
            raise ValueError('Resep/counter checkpoint EMA berbeda atau tidak sah.')
        self.model.load_state_dict(payload['model'], strict=True)
        self.updates = payload['updates']
        self.model.eval().requires_grad_(False)

    def report(self, history, selected_epoch):
        selected = next(row for row in history if row['epoch'] == selected_epoch)
        return {'Jenis Bobot Prediksi': 'ema', 'Batas Decay': self.maximum, 'Offset Warmup': self.offset,
                'Epoch Terpilih': selected_epoch, 'Epoch Dijalankan': len(history),
                'Update EMA Checkpoint Terpilih': selected['ema_updates'],
                'Decay Checkpoint Terpilih': selected['ema_decay'],
                'Update EMA Total Latihan': self.updates,
                'Decay Akhir Latihan': ema_decay_at(self.updates, self.maximum, self.offset),
                'Jumlah Parameter Dirata-rata': sum(p.numel() for p in self.model.parameters()),
                'Parameter EMA Memerlukan Gradien': any(p.requires_grad for p in self.model.parameters())}


def ema_comparison_rows(training):
    """Same-epoch validation diagnostics, not a separately retrained E43 control."""
    rows = []
    for entry in training['history']:
        if 'validation' not in entry:
            continue
        online = entry['validation_online']['RMSE']
        averaged = entry['validation']['RMSE']
        rows.append({'Epoch': entry['epoch'], 'RMSE Online (poin)': online,
                     'RMSE EMA (poin)': averaged, 'Selisih EMA - Online (poin)': averaged-online,
                     'Update EMA': entry['ema_updates'], 'Decay EMA': entry['ema_decay'],
                     'Checkpoint Terpilih': entry['epoch'] == training['best_epoch']})
    return rows


def ema_history_rows(training):
    return [{'Epoch': r['epoch'], 'Update EMA': r['ema_updates'], 'Decay EMA': r['ema_decay'],
             'Langkah Optimizer': r['optimizer_steps'], 'Langkah LR Aktif': r['active_lr_steps'],
             'Langkah AMP Dilewati': r['amp_skipped_steps']} for r in training['history']]


def ema_report_tables(source):
    configurations, comparisons, updates = [], [], []
    for file in sorted(source.files):
        match = re.fullmatch(r'09_S3_fold_(\d+)\.xlsx', file)
        if not match and file != '10_S3_refit.xlsx':
            continue
        prefix = {'Tahap': 'Validasi silang' if match else 'Latihan final',
                  'Fold': int(match[1]) if match else None}
        configurations.extend({**prefix, **row} for row in source.records(file, 'Konfigurasi_EMA'))
        comparisons.extend({**prefix, **row} for row in source.records(file, 'Perbandingan_EMA'))
        updates.extend({**prefix, **row} for row in source.records(file, 'Update_EMA'))
    tables = {}
    for title, rows in [('Konfigurasi EMA', configurations), ('Perbandingan EMA', comparisons), ('Update EMA per Epoch', updates)]:
        if rows:
            keys = list(rows[0])
            tables[title] = Table(keys, [[r[k] for k in keys] for r in rows])
    return tables


def verify_ema_exports(configurations, comparisons, updates, folds, cfg):
    """Validate serialized checkpoint choice, warmup/counts, and online/EMA metrics."""
    validate_ema_config(cfg)
    if len(configurations) != cfg['folds']+1:
        raise ValueError('Konfigurasi EMA checkpoint tidak lengkap.')
    total_comparisons = total_updates = 0
    for fold in [*range(1, cfg['folds']+1), None]:
        stage = 'Latihan final' if fold is None else 'Validasi silang'
        config_rows = [r for r in configurations if r['Fold'] == fold and r['Tahap'] == stage]
        sequence = [r for r in updates if r['Fold'] == fold and r['Tahap'] == stage]
        compare = [r for r in comparisons if r['Fold'] == fold and r['Tahap'] == stage]
        if len(config_rows) != 1 or not sequence:
            raise ValueError('Identitas checkpoint/update EMA hilang atau duplikat.')
        row = config_rows[0]; epochs = row['Epoch Dijalankan']; best = row['Epoch Terpilih']
        if (row['Jenis Bobot Prediksi'] != 'ema' or row['Batas Decay'] != cfg['ema_decay_max']
                or row['Offset Warmup'] != cfg['ema_warmup_offset']
                or row['Parameter EMA Memerlukan Gradien'] is not False
                or not isinstance(row['Jumlah Parameter Dirata-rata'], int) or row['Jumlah Parameter Dirata-rata'] <= 0
                or type(epochs) is not int or type(best) is not int or not 1 <= best <= epochs
                or [r['Epoch'] for r in sequence] != list(range(1, epochs+1))):
            raise ValueError('Resep/epoch/bobot checkpoint EMA tidak sesuai E53.')
        previous = 0
        for update in sequence:
            count = update['Update EMA']
            if (type(count) is not int or count < 1 or count < previous or count != update['Langkah LR Aktif']
                    or count > update['Langkah Optimizer'] or update['Langkah AMP Dilewati'] < 0
                    or not math.isclose(update['Decay EMA'], ema_decay_at(count, cfg['ema_decay_max'], cfg['ema_warmup_offset']), abs_tol=1e-12)):
                raise ValueError('Counter/decay EMA tidak cocok dengan update optimizer yang aktif.')
            previous = count
        selected = sequence[best-1]
        if (row['Update EMA Total Latihan'] != sequence[-1]['Update EMA']
                or row['Update EMA Checkpoint Terpilih'] != selected['Update EMA']
                or not math.isclose(row['Decay Checkpoint Terpilih'], selected['Decay EMA'], abs_tol=1e-12)
                or not math.isclose(row['Decay Akhir Latihan'], sequence[-1]['Decay EMA'], abs_tol=1e-12)):
            raise ValueError('Statistik EMA checkpoint terpilih/akhir latihan berbeda.')
        if fold is None:
            if compare or best != epochs:
                raise ValueError('Refit EMA tidak boleh memilih bobot dari evaluasi test.')
        else:
            if [r['Epoch'] for r in compare] != list(range(1, epochs+1)):
                raise ValueError('Perbandingan EMA/online validasi tidak lengkap.')
            for a, b in zip(compare, sequence):
                if (any(not isinstance(a[k], (int, float)) or not math.isfinite(a[k]) or a[k] < 0
                        for k in ['RMSE Online (poin)', 'RMSE EMA (poin)'])
                        or a['Update EMA'] != b['Update EMA'] or a['Decay EMA'] != b['Decay EMA']
                        or not math.isclose(a['Selisih EMA - Online (poin)'], a['RMSE EMA (poin)']-a['RMSE Online (poin)'], abs_tol=1e-9)
                        or a['Checkpoint Terpilih'] is not (a['Epoch'] == best)):
                    raise ValueError('Perbandingan online/EMA terserialisasi tidak cocok.')
            minimum = min(compare, key=lambda r: r['RMSE EMA (poin)'])
            expected = next(r for r in folds if r['Fold'] == fold)
            if minimum['Epoch'] != best or not math.isclose(minimum['RMSE EMA (poin)'], expected['RMSE (poin)'], rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError('Epoch/metrik fold tidak berasal dari RMSE validasi EMA minimum.')
        total_comparisons += len(compare); total_updates += len(sequence)
    if total_comparisons != len(comparisons) or total_updates != len(updates):
        raise ValueError('Ada baris EMA dari tahap/fold lain.')
    return {'ema_checkpoints': len(configurations), 'ema_comparison_rows': len(comparisons),
            'ema_epoch_update_rows': len(updates), 'ema_selection_counts_and_decay_verified': True}
