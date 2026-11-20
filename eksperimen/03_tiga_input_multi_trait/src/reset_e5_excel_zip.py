def export_excel_zip(path, groups, metadata, max_bytes=30_000_000):
    """Simpan workbook utuh per bagian. Pecah lagi jika ukuran aktual melewati batas."""
    from tempfile import TemporaryDirectory

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inventory = []
    with TemporaryDirectory(prefix='xlsx_', dir=path.parent) as temporary:
        staging = Path(temporary)

        def write_part(stem, tables, relative_part='', row_start=0):
            candidate = staging / f'{stem}{relative_part}.xlsx'
            counts = write_excel(candidate, tables, metadata)
            size = candidate.stat().st_size
            if size < max_bytes:
                inventory.append({
                    'file': candidate.name, 'bytes': size, 'sha256': digest(candidate),
                    'sheet_sizes': counts,
                    'tables': [
                        {'name': name, 'row_start': row_start, 'rows': len(frame)}
                        for name, frame in tables.items()
                    ],
                })
                return

            candidate.unlink()
            if len(tables) > 1:
                entries = list(tables.items())
                middle = len(entries) // 2
                write_part(stem, dict(entries[:middle]), relative_part + '_1', row_start)
                write_part(stem, dict(entries[middle:]), relative_part + '_2', row_start)
                return

            name, frame = next(iter(tables.items()))
            if len(frame) <= 1:
                raise ValueError(f'Sheet {name} tidak dapat dipisah lagi agar di bawah {max_bytes:,} byte.')
            middle = len(frame) // 2
            write_part(stem, {name: frame.iloc[:middle]}, relative_part + '_1', row_start)
            write_part(stem, {name: frame.iloc[middle:]}, relative_part + '_2', row_start + middle)

        for stem, tables in groups.items():
            if tables:
                write_part(stem, tables)

        # XLSX sudah dikompresi. Simpan langsung tanpa mengulang kompresi ZIP.
        pending = staging / 'hasil.zip'
        folder = path.stem
        manifest_name = f'{folder}/manifest.json'
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_STORED) as archive:
            for entry in inventory:
                archive.write(staging / entry['file'], f"{folder}/{entry['file']}")
            archive.writestr(manifest_name, json.dumps({
                'xlsx_size_limit_bytes_exclusive': max_bytes,
                'table_row_start_is_zero_based': True,
                'workbooks': inventory,
            }, ensure_ascii=False, indent=2))

        with zipfile.ZipFile(pending) as archive:
            sizes = {item.filename: item.file_size for item in archive.infolist()}
            expected = {f"{folder}/{entry['file']}": entry['bytes'] for entry in inventory}
            if (set(sizes) != {*expected, manifest_name}
                or any(sizes[name] != size or size >= max_bytes for name, size in expected.items())
                or archive.testzip() is not None):
                raise IOError('Isi ZIP gagal diverifikasi.')
        pending.replace(path)
    return inventory
