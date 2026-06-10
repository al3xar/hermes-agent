"""
Shared platform registry for Hades Agent.

Single source of truth for platform metadata consumed by both
skills_config (label display) and tools_config (default toolset
resolution).  Import ``PLATFORMS`` from here instead of maintaining
duplicate dicts in each module.
"""

from collections import OrderedDict
from typing import NamedTuple


class PlatformInfo(NamedTuple):
    """Metadata for a single platform entry."""
    label: str
    default_toolset: str


# Ordered so that TUI menus are deterministic.
PLATFORMS: OrderedDict[str, PlatformInfo] = OrderedDict([
    ("cli",            PlatformInfo(label="🖥️  CLI",            default_toolset="hades-cli")),
    ("telegram",       PlatformInfo(label="📱 Telegram",        default_toolset="hades-telegram")),
    ("discord",        PlatformInfo(label="💬 Discord",         default_toolset="hades-discord")),
    ("slack",          PlatformInfo(label="💼 Slack",           default_toolset="hades-slack")),
    ("whatsapp",       PlatformInfo(label="📱 WhatsApp",        default_toolset="hades-whatsapp")),
    ("signal",         PlatformInfo(label="📡 Signal",          default_toolset="hades-signal")),
    ("bluebubbles",    PlatformInfo(label="💙 BlueBubbles",     default_toolset="hades-bluebubbles")),
    ("email",          PlatformInfo(label="📧 Email",           default_toolset="hades-email")),
    ("homeassistant",  PlatformInfo(label="🏠 Home Assistant",  default_toolset="hades-homeassistant")),
    ("mattermost",     PlatformInfo(label="💬 Mattermost",      default_toolset="hades-mattermost")),
    ("matrix",         PlatformInfo(label="💬 Matrix",          default_toolset="hades-matrix")),
    ("dingtalk",       PlatformInfo(label="💬 DingTalk",        default_toolset="hades-dingtalk")),
    ("feishu",         PlatformInfo(label="🪽 Feishu",          default_toolset="hades-feishu")),
    ("wecom",          PlatformInfo(label="💬 WeCom",           default_toolset="hades-wecom")),
    ("wecom_callback", PlatformInfo(label="💬 WeCom Callback",  default_toolset="hades-wecom-callback")),
    ("weixin",         PlatformInfo(label="💬 Weixin",          default_toolset="hades-weixin")),
    ("qqbot",          PlatformInfo(label="💬 QQBot",           default_toolset="hades-qqbot")),
    ("yuanbao",        PlatformInfo(label="🤖 Yuanbao",         default_toolset="hades-yuanbao")),
    ("webhook",        PlatformInfo(label="🔗 Webhook",         default_toolset="hades-webhook")),
    ("api_server",     PlatformInfo(label="🌐 API Server",      default_toolset="hades-api-server")),
    ("cron",           PlatformInfo(label="⏰ Cron",            default_toolset="hades-cron")),
])


def platform_label(key: str, default: str = "") -> str:
    """Return the display label for a platform key, or *default*.

    Checks the static PLATFORMS dict first, then the plugin platform
    registry for dynamically registered platforms.
    """
    info = PLATFORMS.get(key)
    if info is not None:
        return info.label
    # Check plugin registry
    try:
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(key)
        if entry:
            return f"{entry.emoji}  {entry.label}" if entry.emoji else entry.label
    except Exception:
        pass
    return default


def get_all_platforms() -> "OrderedDict[str, PlatformInfo]":
    """Return PLATFORMS merged with any plugin-registered platforms.

    Plugin platforms are appended after builtins.  This is the function
    that tools_config and skills_config should use for platform menus.
    """
    merged = OrderedDict(PLATFORMS)
    try:
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if entry.name not in merged:
                merged[entry.name] = PlatformInfo(
                    label=f"{entry.emoji}  {entry.label}" if entry.emoji else entry.label,
                    default_toolset=f"hades-{entry.name}",
                )
    except Exception:
        pass
    return merged
