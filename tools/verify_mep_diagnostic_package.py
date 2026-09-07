#!/usr/bin/env python3
"""Mechanical PDF/source/length parity checks, never semantic correctness."""
import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import sys

import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read,write
from tools.render_mep_diagnostic_package import metric_length
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256


def verify(args):
    manifest=read(args.pdf.with_suffix('.manifest.json'))
    if _file_sha256(args.pdf)!=manifest['pdf_sha256'] or _file_sha256(args.source)!=manifest['source_sha256']:
        raise ValueError('PDF or source hash changed')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with fitz.open(args.pdf) as pdf,fitz.open(args.source) as source:
        if len(pdf)!=len(manifest['pages']):raise ValueError('PDF page count mismatch')
        if manifest['status']!='partial_render_preview' and len(pdf)!=len(source):raise ValueError('original sheets missing')
        for page,record in zip(pdf,manifest['pages']):
            original=source[record['page']-1]
            images=page.get_images()
            if len(images)!=1:raise ValueError('must have one raster base per sheet')
            raw=pdf.extract_image(images[0][0])['image']
            expected=original.get_pixmap(matrix=fitz.Matrix(2,2),colorspace=fitz.csRGB,annots=True).tobytes('jpeg',jpg_quality=88)
            if hashlib.sha256(raw).digest()!=hashlib.sha256(expected).digest():raise ValueError('base differs from original source appearance')
            if fitz.Pixmap(pdf,images[0][0]).colorspace.n!=3:raise ValueError('source colour lost')
            if page.get_image_rects(images[0][0])!=[original.rect]:raise ValueError('source image placement changed')
            for r in record['identified']+record['candidates']:
                length=sum(math.dist(a,b) for a,b in zip(r['polyline_display'],r['polyline_display'][1:]))
                if not math.isclose(length,r['projected_path_display_points'],abs_tol=1e-4,rel_tol=1e-5):
                    if r in record['identified'] or r.get('diagnostic_length_state')!='unknown_reported_and_rendered_length_conflict':
                        raise ValueError('unaccounted length conflict: '+r['id'])
            if 'drawing_inches_per_paper_inch' not in record and manifest['status']!='partial_render_preview':
                raise ValueError('final sheet missing explicit scale state')
            if 'drawing_inches_per_paper_inch' in record:
                lengths=defaultdict(lambda:[0.,0.])
                for group,key,index in ((record['identified'],'system',0),(record['candidates'],'system_hypothesis',1)):
                    for r in group:
                        bucket=lengths[r.get(key)]
                        if r.get('diagnostic_length_state','consistent')!='consistent':bucket[index]=None
                        elif bucket[index] is not None:bucket[index]+=r['projected_path_display_points']
                if {r['system'] for r in record['schedule']}!=set(lengths):raise ValueError('schedule systems missing')
                for r in record['schedule']:
                    expected=[metric_length(v,record['drawing_inches_per_paper_inch']) for v in lengths[r['system']]]
                    if expected!=[r['identified_projected_m'],r['candidate_projected_m']]:raise ValueError('schedule totals do not replay')
            if record['rendered_route_count']!=len(record['identified'])+len(record['candidates']):raise ValueError('route count mismatch')
            if record['unplaced_labels']:raise ValueError('identified routes missing labels')
            if record['installed_length'] is not None or record['purchase_length'] is not None:raise ValueError('quantity authority changed')
            image=args.output/f'page-{record["page"]:03d}.png'
            page.get_pixmap(matrix=fitz.Matrix(1800/page.rect.width,1800/page.rect.width),colorspace=fitz.csRGB).save(image)
            rows.append({'source_page':record['page'],'image':str(image.resolve()),'image_sha256':_file_sha256(image),
                         'original_appearance_replayed':True,'rendered_lengths_replayed':True,'visual_review':'pending'})
            print(record['page'],'source/geometry parity passed',flush=True)
    write(args.output/'checks.json',{'pdf_sha256':manifest['pdf_sha256'],'pages':rows,
          'mechanical_checks_passed':True,'semantic_correctness_established':False,'visual_review':'pending'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'M&P mark-up against shop systems piping.pdf')
    verify(p.parse_args())
