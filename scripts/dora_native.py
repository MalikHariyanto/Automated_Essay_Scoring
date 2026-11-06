"""Visible E52 helpers: official PEFT DoRA, parameter audit, and report checks."""


def validate_dora_config(cfg):
    """E52 changes the encoder training recipe, not the E43 fusion architecture."""
    if cfg['encoder_adaptation'] != 'dora' or cfg['dora_use_dora'] is not True:
        raise ValueError('E52 memerlukan DoRA aktif, bukan LoRA biasa/full fine-tuning.')
    if type(cfg['dora_rank']) is not int or cfg['dora_rank'] != 8:
        raise ValueError('E52 mengunci rank DoRA 8.')
    if cfg['dora_alpha'] != 16 or cfg['dora_dropout'] != 0.0 or cfg['dora_targets'] != ['query', 'value']:
        raise ValueError('E52 mengunci alpha 16, dropout adapter 0, target query/value.')
    if not math.isfinite(cfg['dora_lr']) or cfg['dora_lr'] <= 0:
        raise ValueError('LR DoRA harus positif dan finite.')
    if cfg['s3_encoder_lr'] != cfg['dora_lr'] or cfg['encoder_lr'] != cfg['dora_lr']:
        raise ValueError('LR kelompok encoder E52 adalah LR adapter DoRA; bobot dasar frozen.')


def attach_dora(model, cfg):
    """Attach real PEFT DoRA AFTER E43 initializes all shared model components."""
    validate_dora_config(cfg)
    if model.scenario != 'S3':
        raise ValueError('DoRA E52 hanya dipasang pada RoBERTa S3.')
    expected = 2 * model.roberta.config.num_hidden_layers
    targets = [name for name, layer in model.roberta.named_modules()
               if isinstance(layer, nn.Linear)
               and name.endswith(('.attention.self.query', '.attention.self.value'))]
    if len(targets) != expected:
        raise ValueError('Target query/value tidak lengkap pada seluruh layer RoBERTa.')
    # E43 has already initialized its shared layers. Preserve subsequent dropout RNG.
    with torch.random.fork_rng(devices=[]):
        adapter = LoraConfig(r=cfg['dora_rank'], lora_alpha=cfg['dora_alpha'],
                             target_modules=cfg['dora_targets'], lora_dropout=cfg['dora_dropout'],
                             bias='none', init_lora_weights=True, use_dora=True)
        model.roberta = get_peft_model(model.roberta, adapter)
    trainable = [name for name, p in model.roberta.named_parameters() if p.requires_grad]
    allowed = ('.lora_A.', '.lora_B.', '.lora_magnitude_vector.')
    if not trainable or any(not any(tag in name for tag in allowed) for name in trainable):
        raise RuntimeError('Hanya A/B dan magnitude DoRA boleh dilatih di dalam RoBERTa.')
    report = dora_adapter_report(model)
    if len(report) != expected or any(not row['DoRA Aktif'] for row in report):
        raise RuntimeError('DoRA tidak aktif pada seluruh target; training diblokir.')
    return model


def dora_adapter_report(model):
    """Selected-checkpoint parameter diagnostics; not causal feature importance."""
    rows = []
    layers = model.roberta.config.num_hidden_layers
    for name, module in model.roberta.named_modules():
        if not hasattr(module, 'lora_magnitude_vector') or 'default' not in module.lora_A:
            continue
        match = re.search(r'encoder\.layer\.(\d+)\.attention\.self\.(query|value)$', name)
        if match is None:
            raise ValueError('DoRA terpasang di luar target attention query/value.')
        a = module.lora_A['default'].weight
        b = module.lora_B['default'].weight
        magnitude = module.lora_magnitude_vector['default'].weight
        base = module.get_base_layer()
        rows.append({'Layer ke-': int(match[1])+1, 'Target': match[2], 'Jumlah Layer Encoder': layers,
                     'Dimensi Input': base.in_features, 'Dimensi Output': base.out_features,
                     'Rank': module.r['default'], 'Alpha': module.lora_alpha['default'],
                     'Skala Alpha per Rank': module.scaling['default'],
                     'DoRA Aktif': bool(module.use_dora['default']),
                     'Bobot Dasar Dilatih': any(p.requires_grad for p in base.parameters()),
                     'A Dilatih': a.requires_grad, 'B Dilatih': b.requires_grad,
                     'Magnitude Dilatih': magnitude.requires_grad,
                     'Parameter Adapter': a.numel()+b.numel()+magnitude.numel(),
                     'Norma A': float(a.detach().float().norm().cpu()),
                     'Norma B': float(b.detach().float().norm().cpu()),
                     'Norma Magnitude': float(magnitude.detach().float().norm().cpu())})
    return rows


def dora_parameter_counts(model):
    """Report actual trainable/frozen counts, including frozen original RoBERTa."""
    groups = {}
    for name, p in model.named_parameters():
        if name.startswith('roberta.'):
            if '.lora_magnitude_vector.' in name:
                group = 'DoRA magnitude'
            elif '.lora_A.' in name or '.lora_B.' in name:
                group = 'DoRA arah A/B'
            else:
                group = 'RoBERTa bobot dasar'
        else:
            group = name.split('.')[0]
        row = groups.setdefault(group, {'Bagian': group, 'Parameter Dilatih': 0, 'Parameter Frozen': 0})
        row['Parameter Dilatih' if p.requires_grad else 'Parameter Frozen'] += p.numel()
    return list(groups.values())


def dora_report_tables(source):
    """Rebuild reports from CV/refit snapshots without evaluating test again."""
    modules, counts = [], []
    for file in sorted(source.files):
        match = re.fullmatch(r'09_S3_fold_(\d+)\.xlsx', file)
        if not match and file != '10_S3_refit.xlsx':
            continue
        prefix = {'Tahap': 'Validasi silang' if match else 'Latihan final',
                  'Fold': int(match[1]) if match else None}
        modules.extend({**prefix, **row} for row in source.records(file, 'Adapter_DoRA'))
        counts.extend({**prefix, **row} for row in source.records(file, 'Parameter_DoRA'))
    tables = {}
    for title, rows in [('Adapter DoRA', modules), ('Parameter Dilatih dan Frozen', counts)]:
        if rows:
            keys = list(rows[0])
            tables[title] = Table(keys, [[r[k] for k in keys] for r in rows])
    return tables


def verify_dora_exports(modules, counts, cfg):
    """Reject disabled/missing adapters, trainable base weights, wrong dimensions/counts."""
    validate_dora_config(cfg)
    total = 0
    for fold in [*range(1, cfg['folds']+1), None]:
        stage = 'Latihan final' if fold is None else 'Validasi silang'
        part = [r for r in modules if r['Fold'] == fold and r['Tahap'] == stage]
        summary = [r for r in counts if r['Fold'] == fold and r['Tahap'] == stage]
        if not part or not summary:
            raise ValueError('Laporan adapter/parameter DoRA checkpoint tidak lengkap.')
        layers = part[0]['Jumlah Layer Encoder']
        if type(layers) is not int or layers < cfg['scalar_mix_layers'] or len(part) != 2*layers:
            raise ValueError('Jumlah modul DoRA berbeda dari dua target per layer encoder.')
        if {(r['Layer ke-'], r['Target']) for r in part} != {(i, t) for i in range(1, layers+1) for t in ['query', 'value']}:
            raise ValueError('Identitas target/layer DoRA hilang atau duplikat.')
        for row in part:
            if (row['Jumlah Layer Encoder'] != layers or row['Dimensi Input'] != 768 or row['Dimensi Output'] != 768
                    or row['Rank'] != cfg['dora_rank'] or row['Alpha'] != cfg['dora_alpha']
                    or row['Skala Alpha per Rank'] != cfg['dora_alpha']/cfg['dora_rank']
                    or row['Parameter Adapter'] != 2*768*cfg['dora_rank']+768
                    or row['DoRA Aktif'] is not True or row['Bobot Dasar Dilatih'] is not False
                    or any(row[k] is not True for k in ['A Dilatih', 'B Dilatih', 'Magnitude Dilatih'])):
                raise ValueError('Konfigurasi adapter E52 salah atau bobot dasar ikut dilatih.')
            if any(not isinstance(row[k], (int, float)) or not math.isfinite(row[k]) or row[k] < 0
                   for k in ['Norma A', 'Norma B', 'Norma Magnitude']):
                raise ValueError('Norma parameter DoRA tidak finite.')
        byname = {r['Bagian']: r for r in summary}
        if len(byname) != len(summary) or not {'RoBERTa bobot dasar', 'DoRA arah A/B', 'DoRA magnitude'} <= set(byname):
            raise ValueError('Kelompok parameter DoRA tidak lengkap atau duplikat.')
        if byname['RoBERTa bobot dasar']['Parameter Dilatih'] != 0 or byname['RoBERTa bobot dasar']['Parameter Frozen'] <= 0:
            raise ValueError('Bobot asli RoBERTa E52 harus frozen.')
        if (byname['DoRA arah A/B']['Parameter Dilatih'] != layers*2*2*768*cfg['dora_rank']
                or byname['DoRA magnitude']['Parameter Dilatih'] != layers*2*768
                or byname['DoRA arah A/B']['Parameter Frozen'] != 0 or byname['DoRA magnitude']['Parameter Frozen'] != 0):
            raise ValueError('Jumlah parameter arah/magnitude DoRA tidak cocok.')
        total += len(part)
    if len(modules) != total:
        raise ValueError('Ada baris adapter dari checkpoint lain.')
    return {'dora_module_rows': total, 'dora_checkpoints': cfg['folds']+1,
            'dora_enabled_and_original_roberta_frozen_verified': True}
