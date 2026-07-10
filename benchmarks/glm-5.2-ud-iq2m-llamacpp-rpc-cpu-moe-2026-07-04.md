# GLM-5.2 UD-IQ2_M llama.cpp RPC CPU-MoE bench

Date: 2026-07-04

Status: **not usable for interactive serving under a 30s SLA**.

## Topline

`unsloth/GLM-5.2-GGUF` `UD-IQ2_M` can be loaded with llama.cpp RPC only when
MoE expert weights are kept off GPU with `--cpu-moe`. That makes it a technical
load success, but not a practical serving target on the current gad + M1 Max
pair.

Observed results:

| Run | Result |
|---|---|
| `-ngl all` without `--cpu-moe` | Failed. ROCm0 tried to allocate `142344.78 MiB` and OOMed. |
| `--cpu-moe`, `-n 1` smoke | Loaded and generated 1 token. Prompt eval was about `0.3 tok/s`. |
| `--cpu-moe`, `-n 8` | Still not complete after more than 13 minutes; aborted as non-usable. |
| `--cpu-moe`, `timeout 30`, `-n 1` | Failed SLA. `timeout` exited `124` after `31.12s` while still in model load / RPC allocation. |

Decision: mark this profile as **bench failed** for cloud-murakumo interactive
LLM serving. It can remain as a research artifact, not a scheduler candidate.

## Hardware

Primary node: `gad`

- CPU host memory: `46 GiB`
- Swap: `8 GiB`
- GPU: AMD Radeon 8060S Graphics, `gfx1151`
- ROCm visible VRAM: `49152 MiB`
- ROCm stack: ROCm 7.13
- Disk after model download: about `90G` free

RPC node: `main-2`

- Apple M1 Max
- Unified memory: `32 GiB`
- llama.cpp RPC-visible Metal memory: `25559 MiB`
- Tailscale IP: `100.108.223.94`

## Model

Repository: `unsloth/GLM-5.2-GGUF`

Quantization: `UD-IQ2_M`

Local path on gad:

```text
/home/gad/models/GLM-5.2-UD-IQ2_M/UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf
```

Downloaded shards:

```text
9.0M  GLM-5.2-UD-IQ2_M-00001-of-00006.gguf
46G   GLM-5.2-UD-IQ2_M-00002-of-00006.gguf
46G   GLM-5.2-UD-IQ2_M-00003-of-00006.gguf
46G   GLM-5.2-UD-IQ2_M-00004-of-00006.gguf
46G   GLM-5.2-UD-IQ2_M-00005-of-00006.gguf
40G   GLM-5.2-UD-IQ2_M-00006-of-00006.gguf
```

Total local directory size: `223G`.

## Builds

gad llama.cpp:

```sh
cmake -S . -B build-hip-rpc -G Ninja \
  -DGGML_HIP=ON \
  -DGGML_RPC=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip-rpc --target llama-cli ggml-rpc-server
```

Mac RPC server:

```sh
cmake -S . -B build-metal-rpc \
  -DGGML_METAL=ON \
  -DGGML_RPC=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal-rpc --target ggml-rpc-server
```

RPC device discovery from gad:

```text
ROCm0: Radeon 8060S Graphics (49152 MiB)
RPC0: 100.108.223.94:50052 (25559 MiB)
```

## Commands

Mac RPC server:

```sh
./build-metal-rpc/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -d MTL0
```

Working smoke command:

```sh
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH \
./build-hip-rpc/bin/llama-cli \
  --rpc 100.108.223.94:50052 \
  --device ROCm0,RPC0 \
  -m /home/gad/models/GLM-5.2-UD-IQ2_M/UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf \
  -ngl all \
  --cpu-moe \
  -sm row \
  -ts 48,25 \
  -mg 0 \
  -c 512 \
  -n 1 \
  -p "Write one Clojure form:" \
  -st \
  --no-warmup
```

30s SLA gate:

```sh
/usr/bin/time -v timeout 30 env \
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH \
  ./build-hip-rpc/bin/llama-cli \
    --rpc 100.108.223.94:50052 \
    --device ROCm0,RPC0 \
    -m /home/gad/models/GLM-5.2-UD-IQ2_M/UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf \
    -ngl all \
    --cpu-moe \
    -sm row \
    -ts 48,25 \
    -mg 0 \
    -c 512 \
    -n 1 \
    -p "Write one Clojure form:" \
    -st \
    --no-warmup
```

Result:

```text
exit_code=124
Elapsed (wall clock) time: 0:31.12
Exit status: 124
```

The command was still in model loading / RPC allocation when timeout killed it.
The timeout also caused an expected RPC-side malformed-response/backtrace during
shutdown; this is treated as a cleanup artifact, not the root failure.

## Notes

- Without `--cpu-moe`, llama.cpp tries to place too much model state on ROCm0:
  `allocating 142344.78 MiB on device 0: cudaMalloc failed: out of memory`.
- With `--cpu-moe`, ROCm memory use during load reached roughly `10-13 GiB`, so
  the setting does avoid the large expert allocation.
- The cost is performance: expert paging is dominated by CPU/mmap and swap
  pressure on gad's `46 GiB` host RAM.
- This result should not be advertised as a production route. The next useful
  comparison is REAP50/Q2_K, or a materially larger RAM node.
