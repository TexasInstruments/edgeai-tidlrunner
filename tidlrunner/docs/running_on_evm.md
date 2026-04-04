# Running Models on the EVM

Model **compilation** is performed on a PC (Ubuntu Linux). Once compiled, the resulting artifacts must be made available on the EVM to run inference. This guide covers two ways to do that:

1. [SCP](#option-1-scp) — copy files from PC to EVM over SSH
2. [NFS](#option-2-nfs-mount) — mount the PC directory on the EVM over the network

---

## Prerequisites

### On PC

Compile the model on the PC first. The compiled artifacts will be written to `work_dirs/` by default:

```bash
tidlrunner-cli compile \
    --model_path data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx \
    --target_device AM62A
```

> **Important:** The target_device setting must match the EVM and the version of `tidl-tools` used for compilation must match the SDK version on the EVM. See [setup.md](./setup.md) for details on installing the correct version.

### On EVM

The edgeai-tidlrunner tool files must be accessible on the target EVM and installed using the `./setup_runner_evm.sh`. Then, the model + model-artifacts must be accessible to run inference on the compiled model.


---

## Option 1: Clone with git

Use `git clone` to retrieve the repository (or just the relevant subdirectories) from the network.

### Copy the full repository

```bash
# Run on PC
git clone git@github.com:TexasInstruments/edgeai-tidlrunner.git
```

Setup the tool: 
```bash
./setup_runner_evm.sh
```

This runs:
```bash
pip3 install -e ./tidlrunner[evm]


### Copy the compiled artifacts and models

Once edgeai-tidlrunner is installed on the EVM, you only need to transfer the model artifacts and input data:

```bash
# Run on PC — copy compiled artifacts
scp -r work_dirs/ root@<EVM_IP>:/opt/edgeai-tidlrunner/work_dirs/

# Run on PC — copy input data, model files
scp -r data/ root@<EVM_IP>:/opt/edgeai-tidlrunner/data/
```

### Run inference on the EVM

SSH into the EVM and run inference from the copied directory:

```bash
ssh root@<EVM_IP>
cd /opt/edgeai-tidlrunner
tidlrunner-cli infer \
    --model_path data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx \
    --target_device AM62A
```

See [example_runner_cli_evm.sh](../../examples/example_runner_cli_evm.sh) for more inference examples.

---

## Option 2: NFS Mount

NFS lets the EVM access the PC's filesystem directly over the network — no manual file copying needed. This is convenient during development since changes on the PC are immediately visible on the EVM.

### On PC — export the directory via NFS

1. Install the NFS server:
    ```bash
    sudo apt install nfs-kernel-server
    ```

2. Add an export entry to `/etc/exports`. Replace `<EVM_IP>` with the EVM's IP address and `/path/to/edgeai-tidlrunner` with the actual path:
    ```
    /path/to/edgeai-tidlrunner <EVM_IP>(rw,sync,no_subtree_check,no_root_squash)
    ```

3. Apply the export and start the NFS server:
    ```bash
    sudo exportfs -a
    sudo systemctl restart nfs-kernel-server
    ```

### On EVM — mount the NFS share

1. Create a mount point and mount the share. Replace `<PC_IP>` with the PC's IP address:
    ```bash
    mkdir -p /opt/edgeai-tidlrunner
    mount -t nfs <PC_IP>:/path/to/edgeai-tidlrunner /opt/edgeai-tidlrunner
    ```

2. To make the mount persistent across reboots, add this line to `/etc/fstab` on the EVM:
    ```
    <PC_IP>:/path/to/edgeai-tidlrunner  /opt/edgeai-tidlrunner  nfs  defaults  0  0
    ```

### Run inference on the EVM

With the directory mounted, run inference the same way as on PC:

```bash
cd /opt/edgeai-tidlrunner
tidlrunner-cli infer \
    --model_path data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx \
    --target_device AM62A --data_name random_dataloader
```

---

## Notes

- Replace `AM62A` with your actual target device (e.g., `AM69A`, `TDA4VH`). See [getting_started.md](./getting_started.md) for the full list of supported devices.
- The `tidlrunner-cli infer` command requires compiled artifacts to already exist in `work_dirs/`. If they are missing, run `compile` first on the PC.
- For a full list of inference options, see [command_line_arguments.md](./command_line_arguments.md).
