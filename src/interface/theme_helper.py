"""
Helper pour appliquer les thèmes dynamiquement aux widgets
"""
import customtkinter as ctk
from typing import Dict


def apply_theme_to_widget(widget, theme: Dict):
    """
    Applique un thème à un widget et tous ses enfants récursivement

    Args:
        widget: Le widget à thématiser
        theme: Dictionnaire du thème avec les couleurs
    """
    colors = theme.get('colors', {})

    # Appliquer au widget courant
    try:
        # Frame
        if isinstance(widget, (ctk.CTkFrame, ctk.CTkScrollableFrame)):
            # Déterminer la couleur de fond appropriée
            current_fg = getattr(widget, '_fg_color', None)
            if current_fg:
                # Essayer de mapper l'ancienne couleur vers la nouvelle
                widget.configure(fg_color=colors.get('bg_secondary', colors.get('bg_primary')))

        # Label
        elif isinstance(widget, ctk.CTkLabel):
            widget.configure(text_color=colors.get('text_primary', '#ffffff'))

        # Button
        elif isinstance(widget, ctk.CTkButton):
            widget.configure(
                fg_color=colors.get('accent_primary', '#4a9eff'),
                hover_color=colors.get('accent_secondary', '#00d4ff'),
                text_color=colors.get('text_primary', '#ffffff')
            )

        # Entry
        elif isinstance(widget, ctk.CTkEntry):
            widget.configure(
                fg_color=colors.get('bg_tertiary', '#2a2e4f'),
                border_color=colors.get('border', '#334155'),
                text_color=colors.get('text_primary', '#ffffff'),
                placeholder_text_color=colors.get('text_muted', '#64748b')
            )

        # Switch
        elif isinstance(widget, ctk.CTkSwitch):
            widget.configure(
                fg_color=colors.get('accent_primary', '#4a9eff'),
                progress_color=colors.get('accent_secondary', '#00d4ff'),
                button_color=colors.get('text_primary', '#ffffff'),
                button_hover_color=colors.get('bg_hover', '#2e3256')
            )

        # OptionMenu
        elif isinstance(widget, ctk.CTkOptionMenu):
            widget.configure(
                fg_color=colors.get('bg_tertiary', '#2a2e4f'),
                button_color=colors.get('accent_primary', '#4a9eff'),
                button_hover_color=colors.get('accent_secondary', '#00d4ff'),
                text_color=colors.get('text_primary', '#ffffff'),
                dropdown_fg_color=colors.get('bg_secondary', '#1a1e3f'),
                dropdown_text_color=colors.get('text_primary', '#ffffff')
            )

        # Textbox
        elif isinstance(widget, ctk.CTkTextbox):
            widget.configure(
                fg_color=colors.get('bg_tertiary', '#2a2e4f'),
                border_color=colors.get('border', '#334155'),
                text_color=colors.get('text_primary', '#ffffff')
            )

        # ProgressBar
        elif isinstance(widget, ctk.CTkProgressBar):
            widget.configure(
                fg_color=colors.get('bg_tertiary', '#2a2e4f'),
                progress_color=colors.get('accent_primary', '#4a9eff')
            )

        # Slider
        elif isinstance(widget, ctk.CTkSlider):
            widget.configure(
                fg_color=colors.get('bg_tertiary', '#2a2e4f'),
                progress_color=colors.get('accent_primary', '#4a9eff'),
                button_color=colors.get('accent_secondary', '#00d4ff'),
                button_hover_color=colors.get('accent_primary', '#4a9eff')
            )

        # CheckBox
        elif isinstance(widget, ctk.CTkCheckBox):
            widget.configure(
                fg_color=colors.get('accent_primary', '#4a9eff'),
                hover_color=colors.get('accent_secondary', '#00d4ff'),
                text_color=colors.get('text_primary', '#ffffff'),
                checkmark_color=colors.get('text_primary', '#ffffff')
            )

        # RadioButton
        elif isinstance(widget, ctk.CTkRadioButton):
            widget.configure(
                fg_color=colors.get('accent_primary', '#4a9eff'),
                hover_color=colors.get('accent_secondary', '#00d4ff'),
                text_color=colors.get('text_primary', '#ffffff')
            )

    except Exception as e:
        # Ignorer les erreurs silencieusement
        pass

    # Appliquer récursivement aux enfants
    try:
        if hasattr(widget, 'winfo_children'):
            for child in widget.winfo_children():
                apply_theme_to_widget(child, theme)
    except Exception:
        pass


def get_dynamic_color(widget_type: str, color_key: str, theme: Dict) -> str:
    """
    Obtient une couleur pour un type de widget donné

    Args:
        widget_type: Type de widget ('frame', 'label', 'button', etc.)
        color_key: Clé de couleur ('primary', 'secondary', 'text', etc.)
        theme: Dictionnaire du thème

    Returns:
        Couleur hex
    """
    colors = theme.get('colors', {})

    color_map = {
        'frame': {
            'primary': colors.get('bg_primary', '#0a0e27'),
            'secondary': colors.get('bg_secondary', '#1a1e3f'),
            'tertiary': colors.get('bg_tertiary', '#2a2e4f'),
            'card': colors.get('bg_card', '#1e2346'),
        },
        'text': {
            'primary': colors.get('text_primary', '#ffffff'),
            'secondary': colors.get('text_secondary', '#94a3b8'),
            'muted': colors.get('text_muted', '#64748b'),
        },
        'button': {
            'primary': colors.get('accent_primary', '#4a9eff'),
            'secondary': colors.get('accent_secondary', '#00d4ff'),
            'hover': colors.get('bg_hover', '#2e3256'),
        },
        'status': {
            'success': colors.get('success', '#00ff88'),
            'warning': colors.get('warning', '#ffaa00'),
            'error': colors.get('error', '#ff4444'),
        }
    }

    return color_map.get(widget_type, {}).get(color_key, '#ffffff')


def create_themed_frame(parent, theme: Dict, frame_type: str = 'secondary', **kwargs) -> ctk.CTkFrame:
    """
    Crée un frame avec les couleurs du thème

    Args:
        parent: Widget parent
        theme: Dictionnaire du thème
        frame_type: Type de frame ('primary', 'secondary', 'tertiary', 'card')
        **kwargs: Arguments supplémentaires pour CTkFrame

    Returns:
        CTkFrame thématisé
    """
    colors = theme.get('colors', {})
    color_key = f'bg_{frame_type}'
    fg_color = colors.get(color_key, colors.get('bg_secondary', '#1a1e3f'))

    return ctk.CTkFrame(parent, fg_color=fg_color, **kwargs)


def create_themed_label(parent, theme: Dict, text: str, text_type: str = 'primary', **kwargs) -> ctk.CTkLabel:
    """
    Crée un label avec les couleurs du thème

    Args:
        parent: Widget parent
        theme: Dictionnaire du thème
        text: Texte du label
        text_type: Type de texte ('primary', 'secondary', 'muted')
        **kwargs: Arguments supplémentaires pour CTkLabel

    Returns:
        CTkLabel thématisé
    """
    colors = theme.get('colors', {})
    color_key = f'text_{text_type}'
    text_color = colors.get(color_key, colors.get('text_primary', '#ffffff'))

    return ctk.CTkLabel(parent, text=text, text_color=text_color, **kwargs)


def create_themed_button(parent, theme: Dict, text: str, **kwargs) -> ctk.CTkButton:
    """
    Crée un bouton avec les couleurs du thème

    Args:
        parent: Widget parent
        theme: Dictionnaire du thème
        text: Texte du bouton
        **kwargs: Arguments supplémentaires pour CTkButton

    Returns:
        CTkButton thématisé
    """
    colors = theme.get('colors', {})

    return ctk.CTkButton(
        parent,
        text=text,
        fg_color=colors.get('accent_primary', '#4a9eff'),
        hover_color=colors.get('accent_secondary', '#00d4ff'),
        text_color=colors.get('text_primary', '#ffffff'),
        **kwargs
    )
