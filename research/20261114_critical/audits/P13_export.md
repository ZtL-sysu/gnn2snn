# P13 actual export and accepted specialization

Actual ARC22 programmatic export completed under arc_run_v7; its raw generated Markdown, TeX, bibliography and PDF are retained. The installed ARC defaults to a NeurIPS template, adds a paper checklist and fails to interpret this study's figure/table insertion anchors. Its completeness checker misclassifies the required combined Results heading and reports missing references. Those raw outputs are not accepted deliverables.

The run-local deterministic export adapter `paper/render_tex.py` specializes that generic export to the user contract: anonymous article, exactly six main sections, no invented venue/checklist, all eleven references and four actual scientific figure assets, and four source-bound tables with evidence captions. The accepted Markdown is unchanged from the primary-approved P10 scientific revision. `paper/main.tex`, `paper/references.bib`, `paper/tables` and `figures` are the accepted source set. Source markers are replaced by labeled floats; the generic ARC output is never represented as the final paper.

This is a documented export repair, not credit for a new model call or a claim that the default template was correct. Full accepted-source compilation/visual verification is still pending P15. The actual citation stage reads an accepted-export snapshot, preserving the raw ARC22 outputs.
