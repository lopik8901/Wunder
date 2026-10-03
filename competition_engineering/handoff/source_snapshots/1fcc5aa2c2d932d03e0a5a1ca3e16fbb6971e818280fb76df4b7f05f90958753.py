"""Freeze primary-paper equation verification without rewriting an earlier library."""
import json
import shutil
from competition_engineering import alternative_representation as c
from competition_engineering.pipeline import write_json
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library

def main():
    library=c.ROOT/'competition_engineering/research_cards/v14'
    if not library.exists():
        shutil.copytree(c.ROOT/'competition_engineering/research_cards/v13',library)
        p=library/'legendre_memory_unit.json';card=json.loads(p.read_text())
        card.update(version=2,source_url='https://papers.nips.cc/paper_files/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf',source_locator='Section2, equations1-4 and ZOH discretization; section3 benchmark scope',curator='Codex; primary paper equation and discretization verification',limitations='A fixed linear memory probe omits the learned nonlinear memory/state coupling of the full LMU. Order and window govern frequency capacity; benchmark results do not establish Connectome gains.')
        write_json(p,validate_card(card))
    cards,h=load_library(library);p=c.OUT/'memory_paper_verification.json'
    if not p.exists():
        write_json(p,{'library_version':14,'cards':len(cards),'manifest_sha256':h,'equations':'Continuous A,B match equation2; state update uses equation4, exact ZOH at dt=1/window512. This tests only fixed memory features, not the full nonlinear LMU.'})
        c.event('primary_equations_verified',report_sha256=c.sha(p))

if __name__=='__main__': main()
