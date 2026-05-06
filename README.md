# MFU ADT Program RAG

RAG assistant for Mae Fah Luang University's School of Applied Digital Technology program documents.

The app reads PDF files from:

- `Bachelor Degree/` or the current `Bechelor Degree/`
- `Master Degree/`
- `PhD Degree/`

Install dependencies first:

```powershell
python3 -m pip install -r requirements.txt
```

If PowerShell says `No module named pip`, enable pip first:

```powershell
python3 -m ensurepip --upgrade
python3 -m pip install -r requirements.txt
```

Check that dependencies installed correctly:

```powershell
python3 -c "import requests, gradio, langchain_community; print('dependencies ok')"
```

The `Scripts is not on PATH` warnings from pip are okay for this app because you run it with `python3 mfu_adt_ui.py`.

Run the Gradio UI with:

```powershell
python3 mfu_adt_ui.py
```

Run the local CLI version with:

```powershell
python3 mfu_adt_rag.py
```
