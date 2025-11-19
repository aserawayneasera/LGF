# Quick patch guide

Edits in `train_lgf.py` to make the CLI smoother.

1) Enable the CLI entry point

Uncomment the last line:
```python
if __name__ == "__main__":
    main()
```

2) Fix a print typo in `main()`

Replace the line that references `args.batch-size` with `args.batch_size`.

3) Optional, change the default validation log dir

In `ValidationLogger.__init__`, replace the hardcoded `/nas.dbms/asera/validation_logs`
with something local like `runs/validation_logs`.
