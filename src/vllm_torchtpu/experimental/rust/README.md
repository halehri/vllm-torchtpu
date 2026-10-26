# Serving exports with the Rust runtime

`bin/rvllm.bin` serves a model with rvllm, a native Rust runtime that runs
the artifacts written by the [AOT exporter](../export/) through PJRT, with
no Python in the serving process. It takes the `vllm serve` command line:

```bash
export RVLLM_BINARY=/path/to/rvllm   # or RVLLM_BINARY_URL, see below
src/vllm_torchtpu/experimental/rust/bin/rvllm.bin serve Qwen/Qwen3-0.6B \
  --max-model-len 2048 --port 8000
```

On first use, the script exports the model and then serves it:

1. It exports with `python -m vllm_torchtpu.experimental.export <model>
   --export-mode all` and the model arguments.
2. It serves the artifact with `rvllm serve <artifact>` and the deployment
   flags.

The artifact is cached by model, model arguments and exporter source, so the
next start skips the export. The server speaks the OpenAI-compatible API on
`/v1/completions` and `/v1/chat/completions`.

## Arguments

The script splits the `vllm serve` command line in two:

- **Deployment flags go to rvllm.** These are `--host`, `--port`,
  `--served-model-name`, `--max-queued-requests`, `--shutdown-timeout` and
  `--structured-decoding-mode`. The disaggregation flags `--role`,
  `--kv-port`, `--kv-listen`, `--kv-advertise` and `--kv-producers` go to
  rvllm too, as do `--async-scheduling` and `--no-async-scheduling`.
  `--load-format` goes to both rvllm and the exporter.
- **Everything else goes to the exporter**, which records in the artifact what
  the runtime needs. Examples are `--max-model-len`, `--tensor-parallel-size`,
  `--max-num-seqs`, `--quantization` and `--speculative-config`. The exporter
  ignores options that only configure vLLM's own API server.

Put the model first, as in `vllm serve <model>`. The exporter's options are
not known to the script, so it takes a bare word after an option as that
option's value.

`bench` and `proxy` run the rvllm binary directly:

```bash
rvllm.bin proxy --port 8000 --prefiller-hosts p --prefiller-ports 8100 \
  --decoder-hosts d --decoder-ports 8200
```

## The rvllm binary

The binary is not part of this repository. Point the script to it with one of
these variables:

| Variable | Meaning |
|---|---|
| `RVLLM_BINARY` | A local `rvllm` binary, used as is. |
| `RVLLM_BINARY_URL` | A directory holding `rvllm` and its `metadata.json`, which has a `sha256` entry. It can be `gs://...`, `https://...` or a local path. The binary is checked against the sha256 and cached. |
| `RVLLM_BINARY_SHA256` | The expected sha256, overriding `metadata.json`. |
| `RVLLM_CACHE_DIR` | The cache root for binaries and exports. The default is `~/.cache/rvllm`. |
| `RVLLM_ARTIFACT_DIR` | Serve this exported artifact instead of exporting. |
| `PYTHON` | The interpreter that runs the exporter. The default is `python3`. |

Downloading from `gs://` uses `gcloud storage` with the caller's credentials.
