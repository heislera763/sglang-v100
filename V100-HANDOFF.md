# V100 deployment handoff

Inventory date: 2026-10-03. This machine now has no GPUs; the main server has
both four-GPU baseboards. Migration will use the SATA SSD, without a direct
server-to-server transfer. Pi endpoint changes are deferred. Fan control and
`nvidia-pstated` are already handled on the main server.

## What to carry

- This checkout, including `.git`, uncommitted changes, and benchmark evidence.
- The complete **RadixArk Qwen3.8 Flash Next NVFP4** model, materialized as real
  files. This is the checkpoint used for the lite/NVLink benchmarks.
- Rebuild the uv environment on the destination from the checked-in setup and
  lockfile. Generated environments and build caches are not the deployment source.

Repository: <https://github.com/heislera763/sglang-v100.git>, branch `main`.
Implementation commit: `756feb1a4f4b9bf2a87c9ffc1a5fb114dac9db86`, based on
upstream `bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824`. The subsequent handoff
commit selects the GPU's NUMA node, allows default custom all-reduce, and
records the resulting NVLink measurements. Use the fork's latest `main`,
rather than the implementation commit alone, to include these updates.

The original `~/sglang-V100` fork, `~/sglang-v100-baseline`, Conda environment,
and old launchers are historical comparison tools, not dependencies of lite.
Swift is a separate checkpoint; it is not the model being staged in this handoff.
API credentials stay outside Git and must be supplied separately on the server.

## Source model and capacity

Source: `$HOME/.llama-server/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4`.
Inventory: **419 files, 206 indexed weight shards, 200 symlinks, no missing
indexed shards or broken links**. The complete resolved file payload is
**135,253,622,894 bytes / 125.96 GiB**. A plain directory `du` reported about
96 GiB because some files are symlinks into the external Hugging Face cache.

Use `rsync -aL` below: `-L` follows those links and writes real files, so the
export has no dependency on this machine's cache or old absolute paths.
Allow at least 140 GiB free for the model, workspace and margin. Source files
remain in place until the destination is verified. No compatibility links are
needed. The later deployment path is selected with `MODEL_PATH`.

## SATA SSD inspection

Physical SSD: **T-FORCE 1TB**, serial **TPBF2311150010501911**, currently `/dev/sda`.

| Partition before export | Previous contents |
| --- | --- |
| `/dev/sda1`, 2 GiB | Old FAT32 EFI partition, label `ESP_comp_st`; replaced during export |
| `/dev/sda2`, 951.9 GiB | LUKS2 UUID `f27d6bfd-125d-478e-976e-b36b22ad6621` |
| Unlocked filesystem | Btrfs UUID `046734f5-8d6c-4e28-a567-aab0d4beafa5`, label `comp_staging_sys` |

The user unlocked `/dev/sda2` read-only as `sata-inspect` and mounted Btrfs
top-level subvolume 5 at `/mnt/sata-inspect`. Its only entries were empty
subvolumes `@` (256), `@home` (257), and `@snapshots` (258); `du` reported 16 KiB
for the top level and zero for each subvolume. The large data filesystem is
therefore empty. This does not claim that the separate EFI partition is empty.
The user subsequently authorized reformatting the SSD for portability. The
export script replaces its old EFI/LUKS layout with GPT and one unencrypted
ext4 partition labeled `SGLANG_TRANSFER`. After that, no unlock key is needed.

The SSD was reformatted on 2026-10-03 with the user's authorization. Its new
ext4 filesystem UUID is `783e715c-01d6-43d0-8ea6-76148386457f`.
The export script's final checksum and unmount message confirms completion.

## Format, copy and verify

Run this from the source machine's normal terminal:

```bash
cd "$HOME/my-projects/sglang-v100-lite"
bash v100_lite/export-to-ssd.sh
```

The script identifies the **SATA** SSD by its exact serial and checks its
existing LUKS UUID before formatting (or validates the transfer partition when
resuming an interrupted format). It unmounts the inspection volume, closes
its mapping, creates one GPT/ext4 partition, and mounts it at
`/mnt/sglang-transfer`. Only that identified SSD is reformatted. If rerun after
formatting, the recognized `SGLANG_TRANSFER` filesystem is reused so an
interrupted copy can continue without another format.

The payload is written under `sglang-handoff/`:

- `models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4/`: real model files, using
  `rsync -aL` to materialize the source's HF-cache symlinks.
- `workspace/sglang-v100-lite/`: Git history, current working files and small
  benchmark/profiling artifacts. `.venv`, runtime caches, native build directories
  and private local configuration are excluded. Setup rebuilds generated files.

Source files remain intact. After copying, checksum comparisons verify both the
model and workspace; the script also checks that the model has no symlinks.
It finishes by syncing and unmounting the SSD. A verification failure leaves
it mounted for review. Completion means the terminal prints:
`Checksums passed; SSD unmounted and ready to move.`

On the main server, mount the transported partition by label or its new UUID
and select its model path with `MODEL_PATH`. For example:

```bash
sudo mkdir -p /mnt/sglang-transfer
sudo mount /dev/disk/by-label/SGLANG_TRANSFER /mnt/sglang-transfer
```

Record the filesystem's new UUID with `lsblk -f` for a permanent mount.
The SSD workspace was exported before its final changes were committed and
pushed. Its Git snapshot therefore still has those files as local changes.
For a clean development checkout, clone the fork's latest `main` and retain
the SSD workspace's ignored `artifacts/` directory as benchmark evidence.
The model export is unaffected by the Git snapshot's age.

## Known working software

| Component | Source machine / tested deployment |
| --- | --- |
| OS and kernel | Debian 13; `6.12.107+deb13-amd64` |
| NVIDIA driver | **580.178.04**, `nvidia-kernel-dkms`, proprietary module |
| CUDA compiler toolkit | `/usr/local/cuda-12.9`, nvcc **12.9.41** |
| Python and uv | Python **3.12.12**; uv **0.12.21** at inventory time |
| Torch / packaged CUDA runtime | **2.13.0+cu126** / **12.6.77** |
| Triton / NCCL | **3.7.1** / **2.29.3** |
| Native kernel package | **sglang-kernel 0.4.8+v100**, built by setup |
| TileLang / TVM FFI | **0.1.12** / **apache-tvm-ffi 0.1.11** |
| Transformers / FlashInfer | **5.12.1** / **0.6.18** |
| Build tools | C/C++ compiler, Git, CUDA toolkit, uv; setup supplies CMake/Python build dependencies |
| Host affinity tool | **numactl**, source package version **2.0.19-1** |

The compiler toolkit and Torch's bundled CUDA runtime have different versions
by design. Reproduce the locked cu126 stack first; do not replace it with a
generic latest Torch/CUDA install. V100 needs the proprietary NVIDIA kernel
module; open modules support Turing and newer, not Volta.
[NVIDIA kernel module documentation](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/kernel-modules.html)

Install the appropriate driver/toolkit packages for the main server's OS;
do not copy `/usr/local/cuda`, kernel modules or old APT configuration as an
installation method. Existing compatible installations may already suffice.
NVIDIA Container Toolkit was installed here but is not needed for this
Docker-free deployment. `nvtop` is optional monitoring software.

## Bring up the main server

1. Identify the transported SSD by serial/UUID and mount its ext4 partition.
   Device names such as `/dev/sda` can change on the destination.
2. Put the transported workspace at the chosen permanent project path, or run
   it from the mounted drive. Point `MODEL_PATH` at the materialized model.
3. Inspect `nvidia-smi -L`, `nvidia-smi topo -m`, and `numactl --hardware` before
   selecting GPUs. Each baseboard is an NVLink clique; eight GPUs are not one
   fully connected NVLink group. This launcher still selects **GPUs 0–3, TP4,
   PP1, port 9001**, with automatic affinity to the first GPU's NUMA node.
   Changing the selected quad or endpoint is a later deployment adjustment.
4. Run `bash v100_lite/setup.sh` in the transported checkout to create the uv
   environment and rebuild native/Marlin extensions. It uses pinned sources
   and project-local build paths, with `CUDA_HOME` and `CUTLASS_DIR` overrides.
5. Supply `LLAMA_API_KEY` through an existing private environment or `ENV_FILE`;
   `MODEL_PATH=/path/to/model ./sglang-server.sh` starts the tested profile.
6. Recreate the small user service at the new checkout path if desired. The
   old `sglang-openai-9001.service` was outside Git and referenced
   `/home/alexander/my-projects/sglang-v100-lite/sglang-server.sh`; it used a
   private `EnvironmentFile`, `KillMode=control-group`, `TimeoutStopSec=180`,
   and `Restart=on-failure`. Avoid copying obsolete Swift/fork services or
   their old absolute paths. Pi and VM tunnel retargeting is deferred.
7. Run the existing `v100_lite/tests/smoke.py`, then a bounded benchmark on
   9001. These tests intentionally do not target production port 9000.

Use [V100-LITE.md](V100-LITE.md) for the serving flags, specialized FP16 QSA
cache fix, kernel rationale, original-fork comparison and exact timing contract.
Mainline supports xhigh without a separate xhigh patch. The existing profile
supports text, images, tools, MTP and server-reported streaming PP/TG timings.

The earlier NVLink TP4/MTP medians were 121.90 TG / 3,107.7 PP at 1K and
117.72 TG / 4,052.8 PP at 25K, in tokens/s. A later single-run Gen2/Gen1 check
gave 122.45 / 3,162.5 and 118.88 / 4,379.3 respectively. Old PCIe connections
showed retries and occasional dropouts; these are comparison points, not a
claim of long-term hardware stability or expected results on the new server.

Source-host extras are recorded for completeness: `/etc/modprobe.d/nvidia-p2p.conf`
set `NVreg_EnablePCIeP2P=1`; GRUB included `pci=noacs`,
`pcie_acs_override=downstream,multifunction`, and `acpi_enforce_resources=lax`.
They are machine-specific history, not prerequisites to copy blindly.
`nvidia-pstated` used high graphics clock 1530 MHz, utilization threshold 5,
and 150 iterations before switching; it is already handled on the destination.
Fan and Pi configuration migration are outside this step.

## Deferred work and constraints

- Keep port 9000 reserved for the live deployment; use 9001 for project tests.
  The workload target remains one active request, with xhigh supported natively.
- Revalidate the eight-GPU topology on the main server. Compare TP8 against
  PP2/TP4 across the two NVLink quads; PP plus speculative decoding is not a
  validated deployment profile. The working baseline is TP4/MTP on one quad.
- Qwen's CPU PLE/ngram offload is already part of the launcher. Do not treat
  arbitrary CPU weight offloading as equivalent or assume it is inexpensive.
- The remaining kernel accuracy, QSA/indexer profiling, Torch/CUDA comparison,
  and upstreaming work is listed in V100-LITE.md. Single-request benchmarks
  do not establish long-term stability or model quality.
- Pi/VM endpoint retargeting is deferred. Preserve existing llama.cpp behavior
  when revisiting it, and use server-reported timing metrics rather than
  client elapsed-time estimates. The earlier anomalous "Continue" behavior
  has not been established as a resolved model/integration issue.
- A possible later model is
  [RadixArk/GLM-5.3-Flash-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4).
  Metadata checked on 2026-10-03: 38 weight shards, 202,912,843,096 bytes
  (188.98 GiB). It cannot reside entirely on one 128 GiB quad; eight 32 GiB
  GPUs provide nominal weight headroom, but runtime memory and V100 support
  remain untested. The checkout contains GLM5-Next model code; its KDA/DSA,
  MLA, hyper-connections and draft paths need a separate compatibility audit.
  The model card's Blackwell launch flags are not a V100 configuration.
  No GLM checkpoint has been downloaded or included in this export.
