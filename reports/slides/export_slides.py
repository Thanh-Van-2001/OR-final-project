"""Render the 15 slide sections (published deck) to a 16:9 PDF with local copies of the figures."""
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

import sys

HERE = Path(__file__).resolve().parent
# usage: python export_slides.py <deck_project_dir>  (folder holding deck.json and slides/*.html)
DECK = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'deck' / 'project'
FIG = HERE.parents[1] / 'figures'
OUT = HERE
BLOBS = {'da6c2328eb167b79beb054f92fbd1d79': 'integrated_budget.png', '8f5787a38f3dba7ecbf161d99cf38777': 'm1_hourly_util.png',
         '75295175b70fe348d48ddc341b5aedf9': 'm1_map_pwait.png', '4a811efbac23fba9285cc2fe08c9da8e': 'm2_budget_curve.png',
         '9bcde0ef9144ecd2e6ffb0dcc0871647': 'm2_plan_map.png', 'db508d9dce6a0819f47f948b7a79fb39': 'm3_flows_map.png',
         'b83b538c18cfe6121ecd7baffac73d69': 'm3_guidance_curve.png', '69a1f010cb690704c935689f6c834e94': 'm4_pricing.png',
         'c64a991244ec7748a61096c6aad2cc35': 'm5_sa_convergence.png'}
order = json.loads((DECK / 'deck.json').read_text(encoding='utf-8'))['order']
parts = []
for sid in order:
    h = (DECK / 'slides' / f'{sid}.html').read_text(encoding='utf-8')
    h = re.sub(r'<aside>.*?</aside>', '', h, flags=re.S)
    h = h.replace('style="', 'style="position:relative; width:1920px; height:1080px; box-sizing:border-box; overflow:hidden; ', 1)
    for bid, f in BLOBS.items():
        h = h.replace(f'/_blob/{bid}', (FIG / f).as_uri())
    parts.append(h)
page = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;700&display=swap">
<style>@page {{ size: 1920px 1080px; margin: 0 }} html, body {{ margin: 0; padding: 0 }}
section {{ break-after: page }} h1, h2, h3, p, ul, table {{ margin: 0 }}
table {{ border-collapse: collapse; width: 100% }} th, td {{ padding: 0.35em 0.6em; border-bottom: 1px solid #D5D0C6 }} th {{ font-weight: 700 }}
img {{ display: block }}</style></head><body>{''.join(parts)}</body></html>"""
tmp = OUT / '_render.html'
tmp.write_text(page, encoding='utf-8')
with sync_playwright() as p:
    b = p.chromium.launch(channel='msedge', args=['--allow-file-access-from-files'])
    pg = b.new_page(viewport={'width': 1920, 'height': 1080})
    pg.goto(tmp.as_uri()); pg.wait_for_load_state('networkidle'); pg.evaluate('document.fonts.ready')
    pg.pdf(path=str(OUT / 'OR_Final_Presentation.pdf'), width='1920px', height='1080px', print_background=True)
    b.close()
tmp.unlink()
print('ok', len(order), 'slides')
