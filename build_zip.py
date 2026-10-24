"""Package the project for Web Learning: every file in the repository (tracked or new, minus git-ignored ones)
plus the compiled reports and slides, into OR_Final_Project_RightPlaceRightTime.zip.
Reports are taken from submission/ (copies with student IDs, made by reports/build_reports.py) when present.
Written to a temporary file first and moved into place only when complete."""
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'OR_Final_Project_RightPlaceRightTime.zip'
REQUIRED = ['reports/final/final_report.pdf', 'reports/progress/progress_report.pdf', 'reports/proposal/proposal.pdf',
            'reports/slides/OR_Final_Presentation.pptx', 'reports/slides/OR_Final_Presentation.pdf']

files = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout.split('\n')
EXCLUDE = {'reports/slides/QA_backup.md', 'build_zip.py'}      # team-internal Q&A notes; the packaging script itself
files = sorted({f for f in files if f and (ROOT / f).is_file() and f not in EXCLUDE} | set(REQUIRED))
missing = [f for f in REQUIRED if not (ROOT / f).is_file()]
assert not missing, f'build these first: {missing}'
tmp = OUT.with_suffix('.zip.part')
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in files:
        # submitted copies of the reports carry the student IDs (submission/ is git-ignored)
        sub = ROOT / 'submission' / Path(f).name
        z.write(sub if f in REQUIRED and sub.is_file() else ROOT / f, f)
with zipfile.ZipFile(tmp) as z:
    assert z.testzip() is None
os.replace(tmp, OUT)
print(f'{OUT.name}: {len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB')
