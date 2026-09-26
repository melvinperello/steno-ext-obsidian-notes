"""Base contracts and lifecycle execution context for steno-svc extensions.
This module provides fallback definitions so extensions can be tested and developed standalone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class ExtensionContext:
    """Immutable execution context passed to extensions upon core speech transcription completion."""

    job_id: str
    filename: str
    audio_path: Path
    archive_dir: Path
    srt_path: Path
    srt_content: str
    default_output_dir: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    progress_callback: Optional[Callable[[float, str], None]] = None
    log_callback: Optional[Callable[[str], None]] = None

    def update_progress(self, percent: float, message: str) -> None:
        """Notifies daemon and UI of extension progress (0-100) and status description."""
        if self.progress_callback:
            try:
                self.progress_callback(percent, message)
            except Exception:
                pass

    def log(self, message: str) -> None:
        """Appends a timestamped activity event to the active job log feed."""
        if self.log_callback:
            try:
                self.log_callback(message)
            except Exception:
                pass


class BaseExtension:
    """Abstract base class for all steno-svc extensions."""

    id: str = "base_extension"
    name: str = "Base Extension"
    description: str = ""
    enabled_by_default: bool = True

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        self.config_dir = config_dir or Path(__file__).parent
        self.config: Dict[str, Any] = self.load_config()

    @property
    def config_file(self) -> Path:
        """Tracked default configuration file (template with safe generic values)."""
        return self.config_dir / "config.json"

    @property
    def secured_config_file(self) -> Path:
        """Untracked/gitignored configuration file for user overrides and secrets."""
        return self.config_dir / "secured.config.json"

    def default_config(self) -> Dict[str, Any]:
        """Returns the default configuration dictionary for this extension."""
        return {"enabled": self.enabled_by_default}

    def load_config(self) -> Dict[str, Any]:
        """Loads extension configuration with priority hierarchy:
        1. Default configuration in code
        2. Tracked generic config.json
        3. Untracked secured.config.json (or legacy config.local.json)
        """
        defaults = self.default_config()
        if self.config_file.exists():
            try:
                data = json.loads(self.config_file.read_text(encoding="utf-8"))
                defaults.update(data)
            except Exception:
                pass

        for secured_name in ("secured.config.json", "config.local.json"):
            sec_file = self.config_dir / secured_name
            if sec_file.exists():
                try:
                    data = json.loads(sec_file.read_text(encoding="utf-8"))
                    defaults.update(data)
                    break
                except Exception:
                    pass
        return defaults

    def save_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Updates and persists extension-specific configuration to secured.config.json."""
        self.config.update(updates)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.secured_config_file.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        except Exception:
            pass
        return self.config

    @property
    def is_enabled(self) -> bool:
        return bool(self.config.get("enabled", self.enabled_by_default))

    def set_enabled(self, enabled: bool) -> None:
        self.save_config({"enabled": enabled})

    def on_completion(self, context: ExtensionContext) -> Optional[Dict[str, Any]]:
        """Executes extension logic upon successful completion of core speech transcription."""
        return None
