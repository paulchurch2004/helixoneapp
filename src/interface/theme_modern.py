COLORS = {
    'bg_primary': '#0d1117',
    'bg_secondary': '#161b22',
    'bg_tertiary': '#1e2329',
    'bg_hover': '#2a3038',
    'accent_green': '#00ff88',
    'accent_red': '#ff3860',
    'accent_blue': '#3b82f6',
    'accent_yellow': '#ffaa00',
    'text_primary': '#f0f6fc',
    'text_secondary': '#8b949e',
    'border': '#30363d'
}

def apply_modern_theme():
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")