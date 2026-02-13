"""
Splash screen with loading progress for HelixOne
"""

import threading
from collections.abc import Callable

import customtkinter as ctk


class SplashScreen:
    """
    Splash screen shown during app startup
    """

    def __init__(self, parent: ctk.CTk | None = None):
        """
        Initialize splash screen

        Args:
            parent: Parent window (if None, creates its own Toplevel)
        """
        self.create_own_window = parent is None

        if self.create_own_window:
            self.window = ctk.CTk()
            self.window.geometry("500x300")
            self.window.title("HelixOne")
        else:
            self.window = ctk.CTkToplevel(parent)
            self.window.geometry("500x300")

        # Center window
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

        # Remove window decorations
        self.window.overrideredirect(True)

        # Configure window
        self.window.configure(fg_color="#0a0a0f")

        # Create UI
        self._create_ui()

        # Loading state
        self.current_step = 0
        self.total_steps = 0
        self.is_loading = True

    def _create_ui(self):
        """Create splash screen UI"""
        main_frame = ctk.CTkFrame(
            self.window,
            fg_color="#0a0a0f",
            border_width=2,
            border_color="#6366f1",
            corner_radius=15,
        )
        main_frame.pack(fill="both", expand=True, padx=1, pady=1)

        # Logo / Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="⚡ HelixOne",
            font=("SF Pro Display", 36, "bold"),
            text_color="#6366f1",
        )
        title_label.pack(pady=(40, 10))

        # Subtitle
        subtitle_label = ctk.CTkLabel(
            main_frame,
            text="Formation Trading Intelligente",
            font=("SF Pro Text", 14),
            text_color="#8b8d98",
        )
        subtitle_label.pack(pady=(0, 30))

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            main_frame,
            width=400,
            height=8,
            corner_radius=4,
            progress_color="#6366f1",
            fg_color="#1a1a2e",
        )
        self.progress_bar.pack(pady=(0, 10))
        self.progress_bar.set(0)

        # Status label
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Initialisation...",
            font=("SF Pro Text", 12),
            text_color="#6b6d7a",
        )
        self.status_label.pack(pady=(0, 10))

        # Version label
        try:
            from updater.version import CURRENT_VERSION

            version_text = f"v{CURRENT_VERSION}"
        except ImportError:
            version_text = "v1.0.5"

        version_label = ctk.CTkLabel(
            main_frame,
            text=version_text,
            font=("SF Pro Text", 10),
            text_color="#4a4c5a",
        )
        version_label.pack(side="bottom", pady=10)

    def update_progress(self, current: int, total: int, message: str = ""):
        """
        Update progress bar

        Args:
            current: Current step number
            total: Total number of steps
            message: Status message to display
        """
        if not self.is_loading:
            return

        self.current_step = current
        self.total_steps = total

        # Update progress bar
        progress = current / total if total > 0 else 0
        self.progress_bar.set(progress)

        # Update status message
        if message:
            self.status_label.configure(text=message)

        # Update window
        self.window.update()

    def set_message(self, message: str):
        """Set status message"""
        self.status_label.configure(text=message)
        self.window.update()

    def close(self):
        """Close splash screen"""
        self.is_loading = False
        try:
            self.window.destroy()
        except Exception as e:  # noqa: B110
            print(f"[Splash] Error closing window: {e}")

    def show(self):
        """Show splash screen (blocking if own window)"""
        if self.create_own_window:
            self.window.mainloop()


def show_splash_with_loading(
    loading_func: Callable,
    steps: list[str],
    on_complete: Callable | None = None,
):
    """
    Show splash screen and execute loading function with progress updates

    Args:
        loading_func: Function to execute (should yield step index)
        steps: List of step descriptions
        on_complete: Callback when loading is complete
    """
    splash = SplashScreen()

    def loading_thread():
        try:
            for i, step in enumerate(steps):
                splash.update_progress(i, len(steps), step)
                # Execute loading logic here
                # You can pass callbacks or use a generator
                if callable(loading_func):
                    loading_func(i)

            splash.update_progress(len(steps), len(steps), "Prêt !")

            # Wait a moment before closing
            splash.window.after(500, lambda: splash.close())

            if on_complete:
                splash.window.after(500, on_complete)

        except Exception as e:
            print(f"[Splash] Error during loading: {e}")
            splash.close()

    # Start loading thread
    thread = threading.Thread(target=loading_thread, daemon=True)
    thread.start()

    # Show splash (blocking)
    splash.show()


# Example usage
if __name__ == "__main__":
    import time

    def mock_loading(step):
        """Mock loading function"""
        time.sleep(0.5)

    steps = [
        "Chargement des modules...",
        "Initialisation de l'interface...",
        "Connexion au backend...",
        "Configuration des thèmes...",
        "Chargement des données...",
    ]

    show_splash_with_loading(mock_loading, steps)
