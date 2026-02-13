"""
Monitoring and crash reporting with Sentry
"""

import os

# Sentry DSN - set via environment variable
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
SENTRY_ENABLED = bool(SENTRY_DSN) and not os.getenv("HELIXONE_DEV")


def init_sentry() -> object | None:
    """
    Initialize Sentry for crash reporting
    Returns the sentry_sdk module if successful, None otherwise
    """
    if not SENTRY_ENABLED:
        print("[Monitoring] Sentry disabled (dev mode or no DSN)")
        return None

    try:
        import sentry_sdk
        from sentry_sdk.integrations.threading import ThreadingIntegration
        from updater.version import CURRENT_VERSION

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            # Set traces_sample_rate to 1.0 to capture 100% of transactions for performance monitoring
            traces_sample_rate=0.1,  # 10% of transactions
            # Set profiles_sample_rate to 1.0 to profile 100% of sampled transactions
            profiles_sample_rate=0.1,
            release=f"helixone@{CURRENT_VERSION}",
            environment="production",
            integrations=[
                ThreadingIntegration(propagate_hub=True),
            ],
            # Attach breadcrumbs for better debugging
            attach_stacktrace=True,
            send_default_pii=False,  # Don't send personally identifiable information
            # Ignore common errors
            ignore_errors=[
                KeyboardInterrupt,
                SystemExit,
            ],
        )

        # Set user context (anonymous)
        sentry_sdk.set_user({"id": "anonymous"})

        print(f"[Monitoring] Sentry initialized (release: {CURRENT_VERSION})")
        return sentry_sdk

    except ImportError:
        print("[Monitoring] Sentry SDK not installed, skipping crash reporting")
        return None
    except Exception as e:
        print(f"[Monitoring] Failed to initialize Sentry: {e}")
        return None


def capture_exception(exception: Exception, context: dict = None):
    """
    Capture an exception and send it to Sentry

    Args:
        exception: The exception to capture
        context: Additional context to attach to the error
    """
    if not SENTRY_ENABLED:
        return

    try:
        import sentry_sdk

        if context:
            with sentry_sdk.push_scope() as scope:
                for key, value in context.items():
                    scope.set_context(key, value)
                sentry_sdk.capture_exception(exception)
        else:
            sentry_sdk.capture_exception(exception)

    except ImportError:
        pass
    except Exception as e:
        print(f"[Monitoring] Failed to capture exception: {e}")


def capture_message(message: str, level: str = "info", context: dict = None):
    """
    Capture a message and send it to Sentry

    Args:
        message: The message to capture
        level: Severity level (debug, info, warning, error, fatal)
        context: Additional context to attach
    """
    if not SENTRY_ENABLED:
        return

    try:
        import sentry_sdk

        if context:
            with sentry_sdk.push_scope() as scope:
                for key, value in context.items():
                    scope.set_context(key, value)
                sentry_sdk.capture_message(message, level=level)
        else:
            sentry_sdk.capture_message(message, level=level)

    except ImportError:
        pass
    except Exception as e:
        print(f"[Monitoring] Failed to capture message: {e}")


def add_breadcrumb(message: str, category: str = "default", level: str = "info", data: dict = None):
    """
    Add a breadcrumb for debugging context

    Args:
        message: Breadcrumb message
        category: Category (navigation, ui.click, network, etc.)
        level: Severity level
        data: Additional data
    """
    if not SENTRY_ENABLED:
        return

    try:
        import sentry_sdk

        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {},
        )

    except ImportError:
        pass  # Sentry not installed, skip silently
    except Exception as e:  # noqa: B110
        print(f"[Monitoring] Breadcrumb error: {e}")


def set_user_context(user_id: str = None, email: str = None, username: str = None):
    """
    Set user context for Sentry reports

    Args:
        user_id: User ID (hashed for privacy)
        email: User email (hashed for privacy)
        username: Username
    """
    if not SENTRY_ENABLED:
        return

    try:
        import sentry_sdk

        user_data = {}
        if user_id:
            # Hash user_id for privacy
            import hashlib

            user_data["id"] = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        if email:
            # Hash email for privacy
            import hashlib

            user_data["email_hash"] = hashlib.sha256(email.encode()).hexdigest()[:16]
        if username:
            user_data["username"] = username

        sentry_sdk.set_user(user_data)

    except ImportError:
        pass
    except Exception as e:
        print(f"[Monitoring] Failed to set user context: {e}")


# Initialize Sentry when module is imported (if enabled)
sentry_sdk = init_sentry()
