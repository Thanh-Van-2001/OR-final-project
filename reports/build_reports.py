"""Compile the proposal, progress report and final report twice:
  - public copy (names only, from team.tex) in place, e.g. reports/proposal/proposal.pdf  -> tracked in git
  - submission copy (names + student IDs, from the git-ignored team_private.tex) -> submission/<name>.pdf (git-ignored)
Each copy gets three pdflatex passes (table of contents and references). Usage: python reports/build_reports.py
"""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / 'submission'
TEX = os.environ.get('PDFLATEX', r'C:/Users/Admin/AppData/Roaming/TinyTeX/bin/windows/pdflatex.exe')
REPORTS = ['reports/proposal/proposal', 'reports/progress/progress_report', 'reports/final/final_report']


def compile_(stem: Path, public: bool) -> None:
    src = stem.name + '.tex'
    arg = (r'\def\PublicBuild{1}\input{' + src + '}') if public else src
    for _ in range(3):
        r = subprocess.run([TEX, '-interaction=nonstopmode', '-halt-on-error', '-jobname', stem.name, arg],
                           cwd=stem.parent, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode != 0:
            raise SystemExit(f'pdflatex failed on {src} (public={public}):\n' + r.stdout[-2000:])


SUB.mkdir(exist_ok=True)
private = (ROOT / 'reports' / 'team_private.tex').is_file()
for rel in REPORTS:
    stem = ROOT / rel
    if private:
        compile_(stem, public=False)
        tmp = SUB / (stem.name + '.pdf.part')
        shutil.copyfile(stem.with_suffix('.pdf'), tmp)
        os.replace(tmp, SUB / (stem.name + '.pdf'))
    compile_(stem, public=True)
    print('built', rel + '.pdf', '+ submission copy' if private else '(no team_private.tex: public copy only)')
