# Native draft recovery and scientific corrections

Actual ARC stage 17 completed in `writing/arc_run_v6/stage-17`. It consumed three request-bound responses originally generated live in `arc_run_v4`; the recovery makes no claim of three additional generations. Original identities and request hashes are retained in `native_execution.json`.

The first two runtime attempts stopped at ARC's example simulated-data guard. A later draft was rejected for malformed math serialization and incorrect T2 facts; its inputs and outputs remain in the versioned runtime directories. The v4 third response used uppercase hexadecimal in its receipt. The bridge now compares hexadecimal receipts case-insensitively. These failures are integration failures, not experiment results.

The second v4 writer edited its response after the consumer read it. Exact consumed prose was recovered from the subsequent original bound request, which embeds the previous draft. The v6 trace records the late edit, source hash and original response identity. Current response files are not asserted to equal their earlier consumed bytes. The v5 context-mismatch rejection is preserved.

Raw ARC output is not scientific acceptance. The primary corrected event eligibility, preactivation-versus-event definitions, per-node opportunity budget, first-site versus all-site scope, parameter counts, learning-rate comparisons, hard-search limits, historical objective labels and the distinction between membrane calibration and residual-allocation diagnostics. Raw generated prose remains available; the accepted scientific working copy is `paper/manuscript.md` after primary inspection. All quantitative experiment arrays remain unchanged.

The bridge's three regression tests pass. They check actual stage invocation/context isolation, prohibited response-field rejection, and actual local execution mode. Test-only fixtures are never credited as production stages.
