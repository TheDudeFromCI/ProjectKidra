# Targeted robustness follow-up

## Immutable inputs and job design

Requires the completed `project_kidra.followup` suite and the earlier matrix/temporal packages. Preparation copies four datasets and seven source checkpoints to a separate root. Original roots and code remain unchanged. The new package has its own fingerprint in addition to the original three fingerprints. No model is instantiated by preparation or status commands.

Every job uses the original 991,462-parameter architecture, full-trajectory gradients, ordinary loss weighting, NOOP weight 0.25, and optimizer learning rate 0.0001. Execution is single tick for these causal comparisons. Packed execution remains the preferred production baseline; this suite does not repeat the already matched batching factor.

| Family | Jobs | Initialization and data | Total updates | Added updates |
|---|---:|---|---:|---:|
| `extend` | 3 | Seed-44 tick/global-time/continuous final 50k checkpoints, including optimizer and RNG states; second shuffled epoch | 100k each | 150k |
| `cross` | 4 | Fresh initialization/RNG seeds 42/44 × sample-order seeds 42/44; same training records | 50k each | 200k |
| `repair` | 2 | Seed-42/44 tick final 50k checkpoints; second epoch with prefix augmentation | 100k each | 100k |
| `audit` | 3 | Existing seed-42 tick/global-time 100k checkpoints and seed-43 continuous 50k checkpoint | unchanged | 0 |

Twelve jobs, 450k additional sample updates. No new unique master records. Repair jobs change the input representation of those records while preserving facts and supervised targets. The source for seed-42 repair is the original tick checkpoint already copied into the follow-up suite, not its 100k continuation. Its control is the completed unaugmented 100k baseline. Seed-44 repair and extension fork the same 50k checkpoint and use the same second-epoch sample order.

Crossed seed runs include diagonal repeats because previous seed/device assignments were uneven. Run all four on the same physical GPU. Keeping the entire new suite on one device is the default recommendation for cleaner control. Two or more processes may share that GPU; choose concurrency to fit your hardware. Example:

```bash
python -m project_kidra.robustness prepare --followup runs/followup --root runs/robustness
python -m project_kidra.robustness run --root runs/robustness --slots cuda:0 cuda:0
```

Optional staged execution:

```bash
# Evaluation only:
python -m project_kidra.robustness run --root runs/robustness --slots cuda:0 cuda:0 --only 'audit-*'
# Complete the seed/order test on a single physical GPU:
python -m project_kidra.robustness run --root runs/robustness --slots cuda:0 cuda:0 --only 'cross-*'
# Then execute unfinished work; completed jobs are skipped:
python -m project_kidra.robustness run --root runs/robustness --slots cuda:0 cuda:0
```

The scheduler also accepts `--slots cuda:0 cuda:0 cuda:1 cuda:1`. That trades comparison cleanliness for more devices; device model and effective optimizer learning rates are recorded. The original config object retains its constructor learning-rate defaults, so the new explicit optimizer-rate field is authoritative for the running optimizers. Source checkpoint rates must equal the declared experiment rate.

## Deterministic augmentation and matched orders

Original epoch order is preserved: `Random(order_seed + 1000003 + epoch)` shuffles the same record indices. The new `order_seed` is independent of the initialization/RNG seed. Continuations use epoch 1. Each prefix repair chooses 0, 8, 16, or 24 FILL tokens and prepends that number to both the MAP and RESULT stream. The choice is a deterministic function of sample ID and global sample-update cursor; it does not consume the model's or shuffle's RNG. Resume repeats the same augmented example if an uncheckpointed sample must be replayed.

The transform reconstructs dependent group starts after the longer observations while preserving the original gaps, window assignments, facts, output texts, and control labels. A zero prefix is a genuine unmodified-layout sample for the generated data. This is randomized layout augmentation, not a staged curriculum. It does not teach arbitrary padding locations; those are evaluated separately.

## Evaluation

Every job runs the existing 20 sections plus 12 extra sections (six per IID/OOD split), all on the same evaluation IDs. That is 32 × 2,000 = 64,000 sample-evaluations per job, or 768,000 for the full suite. Evaluation can dominate runtime. No estimate from previous concurrent runs is a reliable completion-time guarantee.

Existing sections retain their meanings: original, clock shift, idle gaps, stagger, reversed tables, prefix24, route change, bounded16; delayed/original; IID/OOD correct-prefix diagnostics; IID exchange ablation.

| Added probe | Change | Interpretation |
|---|---|---|
| `suffix24` | Append 24 FILL tokens to each table | Same added length as prefix24, without moving the original table content |
| `aftermarker24` | Insert 24 FILL tokens after MAP/RESULT, before key/value pairs | Retains marker position while moving the actual lookup content |
| `prefix40` | Prepend 40 FILL tokens to both tables | Longer held-out prefix beyond augmentation support |
| `prefix24_conditional` | prefix24 with correct output prefixes restored between stages | Locate routing vs second-lookup vs answer-stage failure |
| `stagger_conditional` | Staggered arrivals with correct stage prefixes | Locate the continuous model's timing-sensitive failure |
| `rebase_shift4096` | Shift clock by 4096, then subtract the first event's time from all offsets and ends | Input clock-origin normalization control |

Prefix24 is within repair-training coverage; prefix40 and different placements are not. Compare suffix/prefix/after-marker jointly: none alone isolates all timing, attention-length, and position mechanisms. Conditional metrics are teacher-forced stage diagnostics, not free-running agent performance. Their paired exact changes are diagnostic only. The clock-rebase control exactly recovers the original schedule for these zero-origin generated records; it does not demonstrate learned time invariance, remove stagger sensitivity, or test elapsed-time understanding.

All standard rollouts still use the existing fixed external stage/wakeup protocol. Bounded16 removes gold output-length cutoffs for that separate condition. The new layout probes do not turn the harness into a lifecycle-complete agent environment.

## Recovery and output integrity

The scheduler refills available device slots, holds kernel locks, detects surviving workers, bounds retry attempts, and skips completed jobs backed by matching identities and result hashes. Training checkpoints remain at 250-sample intervals unless inherited options differ. Full optimizer states, CPU/device RNGs, accumulated loss, history, and the global sample cursor are saved. A hard crash repeats uncommitted work from the last intact checkpoint; previous-generation checkpoint fallback remains available. Augmentation is cursor-deterministic.

Evaluation checks stopping between records and commits sections atomically. An unfinished section is recomputed; a completed section is reused only if its per-example file hash matches. A resumed completed job is skipped. Source snapshots are immutable copies; no original checkpoint is rotated or repaired. Stop with Ctrl-C and rerun the same scheduler command.

The export command additionally checks that every job completed, that result identities match the manifest, and that every declared per-example section exists with the expected hash and line count. It writes metadata, JSONL evaluation records, histories, environments, and logs; input data and weights stay local. It works for the old follow-up suite too:

```bash
python -m project_kidra.robustness export --root runs/followup --output kidra-followup-complete.tar.gz
python -m project_kidra.robustness export --root runs/robustness --output kidra-robustness-results.tar.gz
```

## Interpretation before changing the architecture

- If seed-44 continuations recover, one epoch was an unreliable stopping rule; report the transition and extra compute rather than claiming a new architecture fixed it.
- If crossed outcomes follow initialization with order held fixed, initialization is the stronger factor among these two seeds. If they follow order, investigate optimization history. Mixed results suggest interaction. Diagonal repeats that disagree with the original runs also require checking hardware/numerical effects.
- If repaired prefixes succeed while original scores stay high, missing training coverage is a sufficient repair for the tested lengths. Stronger claims require held-out placement/length success and more seeds.
- If repair fails, inspect conditional stage scores and compare marker/prefix/suffix probes before changing positional encoding or attention structure.
- If normalized clock origin restores only shifted-clock accuracy, normalize deployment clocks where appropriate; do not claim the architecture handles elapsed time or interleaving generally.

The practical prefix-repair gate is at least 95% exact on original and prefix24 for both IID/OOD. All other probes and every seed remain reportable regardless of that gate. This suite does not establish production-scale reliability, and it does not modify the separate decision-query/lifecycle reference design.
