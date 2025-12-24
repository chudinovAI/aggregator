from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Iterable, Sequence

from .config import AggregatorConfig
from .types import Post

LOGGER = logging.getLogger(__name__)


class EmailNotifier:
    """Send digest summaries via SMTP."""

    def __init__(self, config: AggregatorConfig) -> None:
        self._config = config

    def is_configured(self) -> bool:
        cfg = self._config
        return (
            cfg.email_notifications_enabled
            and cfg.email_smtp_host
            and cfg.email_smtp_username
            and cfg.email_smtp_password
            and cfg.email_sender
            and cfg.email_recipients
        )

    def _build_message(
        self,
        posts: Sequence[Post],
        report_path: str,
        extra_paths: Iterable[str | None],
        subject: str,
        recipients: Sequence[str],
    ) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self._config.email_sender
        message["To"] = ", ".join(recipients)

        top_lines = []
        for index, post in enumerate(posts[:10], start=1):
            title = post.get("title", "") or "Untitled"
            source = post.get("source", "unknown")
            score = float(post.get("combined_score", 0.0))
            url = post.get("url", "")
            top_lines.append(
                f"{index:02d}. [{source}] {title} (score={score:.3f})\n{url}"
            )

        extras = [report_path]
        extras.extend(path for path in extra_paths if path)
        extra_block = "\n".join(extras)

        message.set_content(
            (
                "Weekly digest generated successfully.\n\n"
                "Top posts:\n" + "\n\n".join(top_lines) + "\n\nReports:\n" + extra_block
            )
        )
        return message

    def send_digest(
        self,
        posts: Sequence[Post],
        report_path: str,
        extra_paths: Iterable[str | None] = (),
    ) -> None:
        if not self.is_configured():
            LOGGER.debug("Email notifier not configured; skipping digest email.")
            return

        cfg = self._config
        message = self._build_message(
            posts,
            report_path,
            extra_paths,
            subject="Weekly Technology Digest",
            recipients=cfg.email_recipients,
        )
        self._send(message, cfg.email_recipients)

    def send_personalized(
        self,
        recipient: str,
        posts: Sequence[Post],
        report_path: str,
        extra_paths: Iterable[str | None] = (),
        subject: str = "Your Personalized Digest",
    ) -> None:
        if not self.is_configured():
            LOGGER.debug("Email notifier not configured; skipping digest email.")
            return

        message = self._build_message(
            posts,
            report_path,
            extra_paths,
            subject=subject,
            recipients=[recipient],
        )
        self._send(message, [recipient])

    def _send(self, message: EmailMessage, recipients: Sequence[str]) -> None:
        try:
            cfg = self._config
            with smtplib.SMTP(
                cfg.email_smtp_host, cfg.email_smtp_port, timeout=30
            ) as client:
                client.starttls()
                client.login(cfg.email_smtp_username, cfg.email_smtp_password)
                client.send_message(message)
            LOGGER.info("Sent digest email to %s", ", ".join(recipients))
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to send digest email.")
