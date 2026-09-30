"""Extract official ZIP to an isolated directory; validate paths and CRCs."""
import json
import time
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]/"competition_engineering"
archive=root/"downloads"/"wnn_connectome_starterpack.zip"
destination=(root/"assets").resolve()
destination.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()
with zipfile.ZipFile(archive) as z:
    entries=[]
    for info in z.infolist():
        target=(destination/info.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"unsafe archive member: {info.filename}")
        if target.exists() and not target.is_dir():
            raise FileExistsError(f"refusing to overwrite {target}")
        entries.append({"name":info.filename,"bytes":info.file_size,"crc32":info.CRC})
    print(json.dumps(entries,indent=2),flush=True)
    for info in z.infolist():
        # zipfile validates CRC during the extraction read.
        z.extract(info,destination)
        print(f"extracted {info.filename}",flush=True)
report={"source":"https://files.wundernn.io/wnn_connectome_starterpack.zip",
        "archive_bytes":archive.stat().st_size,"entries":entries,
        "extraction_seconds":time.perf_counter()-started}
(root/"environment"/"download_manifest.json").write_text(json.dumps(report,indent=2))
