# Goalkeeper checkpoint provenance & reproduction

Every command below is the **exact** one used (recovered from the session log),
with the md5 of each artifact so the chain is verifiable. Intermediate
checkpoints/datasets live in `logs/repairs/` (git-ignored, on the WSL training
box). The committed deliverables are the reference teacher `goalkeeper.pt`, the
distilled `goalkeeper_distilled_v3.pt`, the single-MLP `goalkeeper_polished.pt`
(78.4%) / `goalkeeper_polished_v2.pt` (80.5%), and **the final `goalkeeper_moe6.pt`
(91.0%)** — see "Stage D" below.

## The 5-step canonical chain (this is what produced the 78.4% single-MLP policy)

```
goalkeeper.pt  (reference teacher, 72.5%, committed, md5 given by file)
  │  ── Stage A: teacher distillation (DAgger BC) ──
  ▼
src/assets/soccer/weight/goalkeeper_distilled_v3.pt   (student, 65%)   [committed]
  │     scripts/distill_goalkeeper.py  --num-envs 512 --dagger-iters 16 \
  │         --out logs/.../model_distilled_v3.pt   # then copied into weight/
  │
  │  ── Stage B: repair-oracle + DAgger, round 1 ──
  ▼
logs/repairs/clean_r1.pt        (dataset, 153600 pairs, base=v3 student)
        scripts/repair_oracle.py --mode collect \
            --checkpoint src/assets/soccer/weight/goalkeeper_distilled_v3.pt \
            --regions 0 1 2 3 4 5 --G 16 --P 64 --iters 6 --clip 1.0 \
            --knot-span 80 --batches 64 --seed 3 --out logs/repairs/clean_r1.pt
  ▼
logs/repairs/student_clean_r1.pt  ==  logs/repairs/base_r2.pt   (69.6%)
        scripts/distill_repairs.py --data logs/repairs/clean_r1.pt \
            --resume src/assets/soccer/weight/goalkeeper_distilled_v3.pt \
            --epochs 35 --out logs/repairs/student_clean_r1.pt
        cp logs/repairs/student_clean_r1.pt logs/repairs/base_r2.pt
  │
  │  ── Stage B: repair-oracle again to build the BC anchor ──
  ▼
logs/repairs/big_r1.pt          (dataset, 2048 scenarios, base=base_r2, clip 0.7)
        scripts/repair_oracle.py --mode collect --checkpoint logs/repairs/base_r2.pt \
            --regions 0 1 2 3 4 5 --G 16 --P 64 --iters 5 --clip 0.7 \
            --knot-span 80 --batches 128 --seed 8 --out logs/repairs/big_r1.pt
  │
  │  ── Stage C: RL polish (the step that beats the imitation ceiling) ──
  ▼
logs/repairs/polish_clean.pt   ==  src/assets/soccer/weight/goalkeeper_polished.pt  (78.4%)
        scripts/train_polish.py --init logs/repairs/base_r2.pt \
            --bc-data logs/repairs/big_r1.pt --num-envs 1024 --warmup 50 \
            --block-iters 5 --blocks 120 --lr 6e-5 --std 0.08 --bc-coef 1.5 \
            --w-conceded 15 --w-intercept 2 --w-body 0 --w-stop 0 --w-posture 1.0 \
            --out logs/repairs/polish_clean.pt
        cp logs/repairs/polish_clean.pt src/assets/soccer/weight/goalkeeper_polished.pt
```

## Stage D — region mixture-of-experts → the 91.0% deliverable

The single MLP plateaus ~80.5% (`goalkeeper_polished_v2.pt`). From it we RL-polish
**one specialist per region** (`train_polish.py --train-regions <r>`; the Up experts add
the crossing-instant reward `--w-cross`), e.g.:

```
goalkeeper_polished_v2.pt  (80.5%, single MLP, committed)
  │  ── per-region RL polish (one expert each, region order 0..5) ──
  ▼
logs/repairs/{sr0,sr1,sr2c,sr3d,sr4,sr5}.pt
        scripts/train_polish.py --init <polished_v2> --bc-data logs/repairs/big_r1.pt \
            --train-regions <r> --warmup 20 --block-iters 5 --blocks 80 \
            --lr 4e-5 --std 0.08 --bc-coef 0.5 --w-conceded 20 --w-intercept 4 \
            --w-body 1 --w-cross 6 --w-posture 1 --out logs/repairs/sr<r>.pt   # Up uses --w-cross
  │  ── bundle the 6 actor state-dicts + the ballistic-gate config ──
  ▼
src/assets/soccer/weight/goalkeeper_moe6.pt   (91.0%, the deliverable)
        # bundle = {sr:[6 actor-only state-dicts in region order],
        #           z_low=0.85, z_up=1.35, vz_low=-5.0,   # height/side/descent gate
        #           latch_hi=5.0,        # latch the gate as soon as bx<5.0 (routing-timing fix, +4%)
        #           mirror_map="2:3"}    # Right-Up := L/R mirror of Left-Up expert (left-bias fix, +1%)
```

Both `latch_hi` and `mirror_map` are **eval-time** settings baked into the bundle (no
retraining); `scripts/eval_moe6.py` reads them automatically. The bundle is stored
**actor-only** (~39 MB; critic+optimizer stripped — they are never used at inference and
would 4× the file past GitHub's 100 MB limit). Eval: `eval_moe6.py --moe6 goalkeeper_moe6.pt
--seed 2810 --num-envs 256 --batches 16` → 3727/4096 = 91.0% (≈90.5% mean across seeds).

## md5 map (which files are byte-identical copies)

| md5 (prefix) | files | what it is |
|---|---|---|
| `c6bd7e3…` | `base_r2.pt` = `student_clean_r1.pt` | 69.6% student (Stage-B round 1) — the RL-polish **init** |
| `72f5641…` | `polish_clean.pt` = `polish_best.pt` = `BEST_78.pt` = **`src/assets/soccer/weight/goalkeeper_polished.pt`** | **78.4% final policy (deliverable)** |
| `1cb87c8…` | `base_r3.pt` = `student_r2gentle.pt` | a later DAgger variant (NOT on the winning path) |

Datasets are all distinct: `clean_r1 / fix_r2 / clean_r2 / big_r1 / anchor78 / round1`
(`obs(N,960)+act(N,29)+blocked(N)`, N = scenarios×150).

## Side experiments / dead ends (NOT on the winning path — kept for the record)

- `round1.pt` — first oracle collect with **base=teacher** (no `--checkpoint`, clip 1.5, seed 1); its distill (`student_r1.pt`) collapsed to 47% → abandoned (covariate shift). Switched to base=student.
- `fix_r2.pt` / `clean_r2.pt` — round-2 oracle from `base_r2` (seed 5 / seed 4). `fix_r2` adds the "protect-easy" rule (base-blocked scenarios keep residual=0). Distilled variants `student_r2*.pt` plateaued ~68–70% (shared-net interference) → superseded by the RL-polish path.
- `big_c08.pt` (seed 7, base=base_r3), `anchor78.pt` (seed 11, base=polish_best) — anchors for further DAgger-RL rounds that added ≤1% → not used in the deliverable.
- `student_big_scratch.pt` — from-scratch distill into a 2048-wide net → 17% (sparse data) → dead end.
- `resid_r1.pt` — frozen-base + residual-head experiment → 68% → dead end.
- `*smoke*.pt` — tiny smoke-test artifacts, ignore.

## Notes

- Exact bit-reproduction is not expected: the oracle (CEM) and RL polish are
  stochastic, and the physics sim diverges across GPU architectures (verified:
  ball trajectory is identical, robot joints diverge by FP rounding). Re-running
  the chain reproduces the ~78% *result*, not the exact weights.
- Stage-A `goalkeeper_distilled_v3.pt` was produced in an earlier session;
  `--dagger-iters` there was ~16–24 (the checkpoint itself is committed, so this
  step does not need re-running to reproduce the deliverable).
