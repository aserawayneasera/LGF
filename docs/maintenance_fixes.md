# Maintenance changes after source recovery

## Shared test exporter: packaged SE/CBAM checkpoints

File: `experiments/paper1_test_evaluation_v1/models.py`.

The recovered shared exporter rebuilt runner-format SE/CBAM checkpoints by
passing only the method name to `core_bridge.build_model`. The archived bridge
selects its baseline configuration for those two names unless an explicit
`original_config` is supplied. Loading the saved attention state then fails
under strict checkpoint loading. The separate
`paper1_se_cbam_extension_nas_v1_0_2/evaluate_test_se_cbam.py` avoided this issue
by installing the attention-aware bridge first.

The shared exporter now checks the stored attention `block_type` against the
requested method and passes the checkpoint's stored configuration into the
unchanged archived bridge. Model definitions, learned tensors, preprocessing,
prediction settings, and scientific training source remain unchanged. Other
runner-format checkpoint methods retain their original construction path.

The unchanged recovered exporter is preserved at
`provenance/archived_source/models_test_export_v1.py.txt`. The text suffix keeps
the historical reference outside recursive Python source fingerprints.

Validation: four dependency-free regression tests in
`experiments/paper1_test_evaluation_v1/tests/test_attention_adapter.py` verify
configuration forwarding, rejection of mismatched attention metadata,
checkpoint identity checks, and unchanged builder arguments for other methods.
These tests isolate adapter arguments and strict state loading with mocks;
they do not claim a real detector inference run.

```bash
python -m unittest discover \
  -s experiments/paper1_test_evaluation_v1/tests \
  -p test_attention_adapter.py -v
```

Changing this adapter changes the source fingerprint of new test exports.
Existing exports retain their original code identity and should be checked with
the source version recorded in their provenance. Do not resume an old export
using a different adapter fingerprint.
