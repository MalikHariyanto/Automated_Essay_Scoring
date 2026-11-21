def pack_excel_zip_parts(path, staging, inventory, max_bytes):
    """Publish independently readable ZIP parts with their own workbook manifest."""
    if not inventory:
        raise ValueError('Tidak ada workbook untuk diekspor.')

    def write_archive(entries, name):
        folder = Path(name).stem
        pending = staging / name
        manifest_name = f'{folder}/manifest.json'
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_STORED) as archive:
            for entry in entries:
                archive.write(staging / entry['file'], f"{folder}/{entry['file']}")
            archive.writestr(manifest_name, json.dumps({
                'zip_size_limit_bytes_exclusive': max_bytes,
                'xlsx_size_limit_bytes_exclusive': max_bytes,
                'table_row_start_is_zero_based': True,
                'workbooks': entries,
            }, ensure_ascii=False, indent=2))
        return pending

    # Split an archive based on its actual size, including ZIP headers and manifest.
    archives = []

    def prepare(entries, suffix=''):
        pending = (write_archive(entries, f'{path.stem}{suffix}.zip')
                   if sum(entry['bytes'] for entry in entries) < max_bytes else None)
        if pending is None or pending.stat().st_size >= max_bytes:
            if pending is not None:
                pending.unlink()
            if len(entries) == 1:
                raise ValueError(f"ZIP {entries[0]['file']} melebihi batas {max_bytes:,} byte.")
            middle = len(entries) // 2
            prepare(entries[:middle], suffix + '_1')
            prepare(entries[middle:], suffix + '_2')
            return
        prefix = pending.stem
        with zipfile.ZipFile(pending) as archive:
            expected = {f"{prefix}/{entry['file']}": entry['bytes'] for entry in entries}
            sizes = {member.filename: member.file_size for member in archive.infolist()}
            if (set(sizes) != {*expected, f'{prefix}/manifest.json'}
                or any(sizes[name] != size or size >= max_bytes for name, size in expected.items())
                or archive.testzip() is not None):
                raise IOError('Isi ZIP gagal diverifikasi.')
        archives.append((pending, entries))

    prepare(inventory)
    published = []
    for pending, entries in archives:
        final = pending.replace(path.parent / pending.name)
        published.append((final, final.stat().st_size, entries))
    return published
