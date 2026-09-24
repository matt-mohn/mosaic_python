"""Windows file picker with explicit cancellation and UTF-8 path handling."""

import base64
import json
import ntpath
import subprocess


def windows_file_dialog(*, save=False, title, file_filter, initial_dir,
                        default_name="", extension=""):
    """Return an accepted absolute path, or an empty string on Cancel.

    Paths travel as encoded JSON data; they are never PowerShell source.
    Process and decoding failures propagate so callers can report them.
    """
    options = base64.b64encode(json.dumps({
        "title": title, "filter": file_filter, "directory": str(initial_dir),
        "name": default_name, "extension": extension,
    }, ensure_ascii=False).encode("utf-8")).decode("ascii")
    kind = "SaveFileDialog" if save else "OpenFileDialog"
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; "
        f"$json = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{options}')); "
        "$cfg = ConvertFrom-Json $json; "
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$owner = New-Object System.Windows.Forms.Form; $owner.TopMost = $true; "
        f"$dialog = New-Object System.Windows.Forms.{kind}; "
        "try { $dialog.Title = $cfg.title; $dialog.Filter = $cfg.filter; "
        "$dialog.InitialDirectory = $cfg.directory; $dialog.FileName = $cfg.name; "
        "$dialog.DefaultExt = $cfg.extension; $dialog.AddExtension = $true; "
        "if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) "
        "{ [Console]::Write($dialog.FileName) } "
        "} finally { $dialog.Dispose(); $owner.Dispose() }"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-STA", "-EncodedCommand", encoded],
        capture_output=True, encoding="utf-8", check=True, timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    path = result.stdout
    if path and not ntpath.isabs(path):
        raise ValueError("File picker returned an invalid path")
    return path
