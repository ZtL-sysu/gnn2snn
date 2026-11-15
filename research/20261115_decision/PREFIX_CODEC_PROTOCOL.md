# Intrinsic local temporal codeword diagnostic — pre-run

The prior spatial2-event/8-candidate learned-selector family is closed. This is a materially different **temporal codeword** hypothesis: directly construct local source-state equivalence rather than predict unknown future full-network costs. No prior codebook/MSE-weight/temperature search is reopened. No method success or novelty is assumed.

## Exact source condition to verify

At the firstGCN position, active tick2 injected current is zero:raw event and embedding bias are absent, then folded conv bias is subtracted. Aftertick1 letq>0 be the original event, ands the actual source state. The known-zero-input tick2 membrane is `s.mem+s.exp_in-s.exp_out`;changingq byδ updatesmem by−δ,exp_out by+δ,so algebraically the next membrane shifts by−2δ (floating point requires exact source validation).

A redundant pair can exist:θ=1,input=.8,initial0 gives(q1,q2)=(1,−.5). Forceq1'=.5 with consistent state updates;the natural source tick2 then givesq2'=+.5. Both decodeq1+q2/2=.75;both have2 nonzero events,while local amplitude sumdrops1.5→1. This example has been executed against the frozen core;not a modified neuron rule.

Eligibility must be checked by the **actual source** on cloned local states withzero next input,not by an approximate algebraic inequality:halfq remains in original8-level alphabet,q2old=−q/2,q2new=+q/2,andT2 local decode is bitwise equal. Keep inputs/state unmodified during probing. Real execution changesonlytick1 event andupdatesallthree recurrentstate fields;actualtick2 is never forced. A local zero-input source probe is extra encoding arithmetic,not free,andnot a full-network/teacher lookahead. Chargeit separately ifa method isadmitted.

This guarantees only first-site localT2 decode/event/amplitude properties. Downstream nonlinear temporal dynamics,classification,total events andruntime are **not** guaranteed. Retain signed,multilevel DCGS semantics;not binaryIFspikes.

## Bounded diagnostic

Teacher41,both tasks,strongsharedT1local.lr.01 baseline. First128 fixed training indices,batch32,no val/test load,no fitting/newteacher. Policies:unchangedbaseline,all eligible coordinates,and deterministicgraph-ID-seeded half ofeligible coordinates(seed20261119). The half policy is a generic partial-applicationcontrol,not graph-aware innovation. No ratio sweep or selectedchannel tuning. Same rules/parameters atT1/T2. Preserve all outputlogits,actualevents,amplitude,branch-deliveryproxy,first-site errors/coverage andsource snapshots. Additionalcap5 wall minutes,serialGPU.

Before any encoder learning:verify realGCN site0 tick2 current exactlyzero onall128 graphs;sourceprimitiveCPUtests andGPUfirst-site decode equality;occupancy eligibility at least1% ofnonzero first-site coordinates ineach task. Ifcoverage/equivalence fails,stop temporalcandidate development. Ifpassed,inspect actualclassification/prefix/cost effects todecidewhether a newminimal same-capacity structure/decision encoding comparison isjustified. Merelyfinding redundant signed-digit words orpreserving a localdecode isnot an algorithmic contribution.

No threshold/offset recalibration,labelsupervision or source repair. No broad searchafter failure. Source recurrence/zero-input assumption andnovelty mustbe checkedbefore graph-specific method claims.

## Pre-learning orientation completion (after first diagnostic, before reverse outputs)

Halve diagnostic passes source equality/coverage onboth tasks,but lowerslocalamplitude while raisingwhole-network events/amplitude,especiallyT2. This is a newlocal-versus-global cost mismatch,not a successfulmethod. Complete the **same two-codeword equivalence class** by checkingthe reverseorientation:original(q1,q2)=(a,+a) mayallow(2a,−a),sameT2decode1.5a andsame2localevents. Its localamplitude increases,so it cannotbe presented as free. Testexactsourceeligibility/rangeand actualfull-network costs before anylearning orvalidation. This is onepredeclared reverseorientation,no newratio/threshold/channel tuning. Reusefirst128 traininggraphs,both tasks,baseline/all/half policies,5-minute cap. Preserveprevioushalveoutputs;reverseoutputs goinnewdirectories. A futurelearnedencoder,ifjustified,maychooseonlythe twoverifiedlegalrepresentations percoordinate,not a larger codebook.
