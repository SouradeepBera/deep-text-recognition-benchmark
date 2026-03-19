# Profiling Changes in `test.py`

This repository now profiles OCR inference more accurately for test-set evaluation.

## What Was Changed

- Moved `prof.step()` so it runs once per evaluation batch instead of once per decoded sample.
- Added CUDA synchronization around the forward pass timing so reported inference time is accurate on GPU.
- Added explicit peak CUDA memory reporting with:
  - `torch.cuda.max_memory_allocated()`
  - `torch.cuda.max_memory_reserved()`
- Added a JSON metadata artifact at `./result/<exp_name>/profile_metadata.json` to store:
  - profiler activities
  - schedule configuration
  - whether memory and FLOPs were enabled
  - peak CUDA memory when available
- Kept the Chrome trace export and CSV summary export for downstream inspection.

## Why The Old Behavior Was Problematic

Previously, `prof.step()` was called inside the loop that computes metrics for each prediction in a batch. That made the profiler schedule advance once per decoded sample rather than once per model iteration.

With a schedule like:

- `wait=1`
- `warmup=2`
- `active=5`

the active profiling window could be consumed during the first batch, especially with large batch sizes. That misaligned the profile with actual model inference and mixed in a lot of string decoding, edit-distance computation, and other Python-side evaluation work.

## What The New Behavior Means

Now each profiler step corresponds to one batch through the model, which is the correct unit for profiling test-set inference.

This makes the trace and summary much more representative of:

- model forward compute
- operator-level memory activity
- supported-operator FLOP estimates

## Important Limitations

- `torch.profiler(with_flops=True)` does **not** guarantee complete FLOP coverage for all operators.
- This OCR model uses convolution, linear, recurrent, and attention-style operations. Some of the recurrent and attention-related work may not be fully reflected in profiler FLOP totals.
- Peak CUDA memory is measured during inference under `torch.no_grad()`, so it does **not** represent training memory usage.
- If running on CPU only, CUDA peak memory fields will naturally be absent.

## Output Files

After evaluation, the profiler writes:

- `./result/<exp_name>/profile_trace.json`
- `./result/<exp_name>/profile_summary.csv`
- `./result/<exp_name>/profile_metadata.json`

## Recommended Interpretation

- Use `profile_trace.json` to inspect timeline behavior and kernel/operator ordering.
- Use `profile_metadata.json` to read peak GPU memory quickly.
- Treat FLOP totals as partial estimates unless you independently validate coverage for the exact OCR architecture you are using.
