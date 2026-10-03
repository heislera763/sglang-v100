#!/usr/bin/env bash
# Authorized physical handoff: format only the identified T-FORCE SSD, then copy.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_model=${MODEL_PATH:-$HOME/.llama-server/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4}
ssd_serial=TPBF2311150010501911
ssd_device=$(lsblk -dnpo PATH,SERIAL | awk -v serial="$ssd_serial" '$2 == serial {print $1}')
[[ -n $ssd_device && $ssd_device != *$'\n'* ]] || { echo 'Expected SSD serial not uniquely found' >&2; exit 1; }
[[ -b $ssd_device ]] || { echo "Cannot access $ssd_device; run from the normal host terminal" >&2; exit 1; }
[[ $(lsblk -dno TRAN "$ssd_device") == sata ]] || { echo 'Expected a SATA device' >&2; exit 1; }
[[ -f $source_model/model.safetensors.index.json ]] || { echo 'Source model not found' >&2; exit 1; }
python3 - "$source_model" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
index = json.loads((root / 'model.safetensors.index.json').read_text())
missing = [name for name in set(index['weight_map'].values()) if not (root / name).is_file()]
if missing:
    raise SystemExit(f'Missing model shards: {missing}')
if any(p.is_symlink() and not p.exists() for p in root.rglob('*')):
    raise SystemExit('Source model contains broken links')
PY
partition=${ssd_device}1
mount_dir=/mnt/sglang-transfer
transfer_root=$mount_dir/sglang-handoff
if [[ $(lsblk -dno FSTYPE "$partition") == ext4 && $(lsblk -dno LABEL "$partition") == SGLANG_TRANSFER ]]; then
    echo 'Existing SGLANG_TRANSFER filesystem found; continuing without formatting'
else
    sudo true
    original_layout=false
    if [[ -b ${ssd_device}2 && $(lsblk -dno UUID "${ssd_device}2") == f27d6bfd-125d-478e-976e-b36b22ad6621 ]]; then
        original_layout=true
    else
        # A previous run may have written GPT before BLKRRPART returned EBUSY.
        [[ $(lsblk -dno PARTLABEL "$partition") == sglang-transfer &&
           $(lsblk -dno PARTTYPE "$partition") == 0fc63daf-8483-4772-8e79-3d69d8477de4 &&
           $(lsblk -nro TYPE "$ssd_device" | awk '$1 == "part" {n++} END {print n+0}') == 1 &&
           $(lsblk -bdno SIZE "$partition") -gt 900000000000 &&
           ( $(lsblk -dno UUID "$partition") == D2C8-5494 || -z $(lsblk -dno FSTYPE "$partition") ) ]] || {
            echo 'Unexpected disk layout; refusing to format' >&2; exit 1;
        }
        echo 'Resuming interrupted format of the identified transfer partition'
    fi
    echo "Formatting $ssd_device, serial $ssd_serial, as GPT + unencrypted ext4"
    if mountpoint -q /mnt/sata-inspect; then
        [[ $(findmnt -n -o SOURCE /mnt/sata-inspect) == /dev/mapper/sata-inspect ]] || { echo 'Unexpected inspection mount' >&2; exit 1; }
        sudo umount /mnt/sata-inspect
    fi
    if [[ -e /dev/mapper/sata-inspect ]]; then
        mapped_device=$(sudo cryptsetup status sata-inspect | awk '$1 == "device:" {print $2}')
        [[ $mapped_device == "${ssd_device}2" ]] || { echo 'Unexpected encrypted mapping' >&2; exit 1; }
        sudo cryptsetup close sata-inspect
    fi
    [[ -z $(lsblk -nro MOUNTPOINTS "$ssd_device" | tr -d '[:space:]') ]] || { echo 'SSD still has mounted filesystems' >&2; exit 1; }
    [[ $(lsblk -nro TYPE "$ssd_device" | awk '$1 != "disk" && $1 != "part"') == '' ]] || { echo 'SSD still has active mappings' >&2; exit 1; }
    if $original_layout; then
        sudo /usr/sbin/sgdisk --zap-all "$ssd_device"
        sudo /usr/sbin/sgdisk --clear --new=1:0:0 --typecode=1:8300 --change-name=1:sglang-transfer "$ssd_device"
        if ! sudo /usr/sbin/blockdev --rereadpt "$ssd_device"; then
            echo 'Partition table written, but kernel reload failed. Run sudo udevadm settle and retry; if still blocked, reboot and retry.' >&2
            exit 1
        fi
    fi
    sudo udevadm settle
    sudo /usr/sbin/mkfs.ext4 -F -m 0 -L SGLANG_TRANSFER "$partition"
fi
sudo mkdir -p "$mount_dir"
if ! mountpoint -q "$mount_dir"; then sudo mount "$partition" "$mount_dir"; fi
[[ $(findmnt -n -o SOURCE "$mount_dir") == "$partition" && $(findmnt -n -o FSTYPE "$mount_dir") == ext4 ]] || { echo 'Unexpected destination mount' >&2; exit 1; }
sudo install -d -o "$(id -u)" -g "$(id -g)" "$transfer_root"
destination_model=$transfer_root/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4
destination_repo=$transfer_root/workspace/sglang-v100-lite
mkdir -p "$destination_model" "$destination_repo"
repo_excludes=(--exclude='/.venv/' --exclude='/.cache/' --exclude='/artifacts/aot-build/'
    --exclude='/artifacts/marlin-v100/' --exclude='/artifacts/cutlass/'
    --exclude='__pycache__/' --exclude='/.aws/' --exclude='/.codex/' --exclude='/.env')
rsync -aL --partial --info=progress2 "$source_model/" "$destination_model/"
rsync -a --partial --info=progress2 "${repo_excludes[@]}" "$repo_root/" "$destination_repo/"
verify_file=$(mktemp)
trap 'rm -f "$verify_file"' EXIT
echo 'Verifying model contents and workspace with checksums'
rsync -aL --checksum --dry-run --itemize-changes "$source_model/" "$destination_model/" > "$verify_file"
rsync -a --checksum --dry-run --itemize-changes "${repo_excludes[@]}" "$repo_root/" "$destination_repo/" >> "$verify_file"
[[ ! -s $verify_file ]] || { cat "$verify_file"; echo 'Copy differs; leaving SSD mounted for review' >&2; exit 1; }
[[ -z $(find "$destination_model" -type l -print -quit) ]] || { echo 'Model export contains a symlink' >&2; exit 1; }
git -C "$destination_repo" status --short
sync
sudo umount "$mount_dir"
echo 'Checksums passed; SSD unmounted and ready to move. Source files remain intact.'
