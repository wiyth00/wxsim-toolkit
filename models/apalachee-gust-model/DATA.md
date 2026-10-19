# Apalachee gust model: full data archive

This folder tracks the code, docs, tables and preview maps only. The full
working folder (3.3 GB, including the `fields.npz` / `frames.npz` outputs,
rendered frames and animations) is attached to the GitHub release
[`apalachee-gust-model-data-v1`](https://github.com/wiyth00/wxsim-toolkit/releases/tag/apalachee-gust-model-data-v1)
as a split tar archive.

## Restore

```bash
gh release download apalachee-gust-model-data-v1 -R wiyth00/wxsim-toolkit
shasum -a 256 -c SHA256SUMS.txt
cat apalachee-gust-model-full.tar.part-* | tar -xf -
```

This recreates `apalachee-gust-model/` exactly as it was on 2026-10-18.

## Checksums

```
cafabd13f74ed446351bf52374ced7b7426edf6614b5b8948556699312a3fcf9  apalachee-gust-model-full.tar.part-aa
6d94b8e561d28f1c9f435072a4f59e0a2235954b7101a10ccd6f0a8ef9375d44  apalachee-gust-model-full.tar.part-ab
```
