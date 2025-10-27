"""
Effets glassmorphism pour HelixOne
Frames, panels, cards avec effet de verre
"""

import customtkinter as ctk
from typing import Optional

class GlassFrame(ctk.CTkFrame):
    """Frame avec effet glassmorphism"""
    
    def __init__(self, parent, border_glow: bool = False, **kwargs):
        # Couleurs par défaut pour effet verre
        default_fg = kwargs.pop('fg_color', '#1e232980')
        
        super().__init__(
            parent,
            fg_color=default_fg,
            corner_radius=15,
            border_width=1,
            border_color='#ffffff20',
            **kwargs
        )
        
        if border_glow:
            self.configure(border_color='#3b82f640')


class GlassPanel(ctk.CTkFrame):
    """Panel glassmorphism avec titre"""
    
    def __init__(self, parent, title: str = "", **kwargs):
        super().__init__(
            parent,
            fg_color='#1e232980',
            corner_radius=15,
            border_width=1,
            border_color='#ffffff20',
            **kwargs
        )
        
        # Header
        if title:
            header = ctk.CTkFrame(self, fg_color='transparent')
            header.pack(fill='x', padx=20, pady=(20, 10))
            
            ctk.CTkLabel(
                header,
                text=title,
                font=('Helvetica', 18, 'bold'),
                text_color='#f0f6fc'
            ).pack(anchor='w')
            
            # Séparateur
            separator = ctk.CTkFrame(self, height=1, fg_color='#ffffff20')
            separator.pack(fill='x', padx=20, pady=(0, 10))
        
        # Conteneur pour le contenu
        self.content = ctk.CTkFrame(self, fg_color='transparent')
        self.content.pack(fill='both', expand=True, padx=20, pady=20)


class GlassCard(ctk.CTkFrame):
    """Carte glassmorphism pour métriques"""
    
    def __init__(self, parent, title: str, value: str, icon: str = "", 
                 trend: str = "", **kwargs):
        super().__init__(
            parent,
            fg_color='#1e232980',
            corner_radius=15,
            border_width=1,
            border_color='#ffffff20',
            width=200,
            height=150,
            **kwargs
        )
        
        self.pack_propagate(False)
        
        # Conteneur principal
        container = ctk.CTkFrame(self, fg_color='transparent')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Icône
        if icon:
            icon_label = ctk.CTkLabel(
                container,
                text=icon,
                font=('Helvetica', 32)
            )
            icon_label.pack(anchor='w', pady=(0, 10))
        
        # Titre
        title_label = ctk.CTkLabel(
            container,
            text=title,
            font=('Helvetica', 12),
            text_color='#8b949e'
        )
        title_label.pack(anchor='w')
        
        # Valeur
        value_label = ctk.CTkLabel(
            container,
            text=value,
            font=('Helvetica', 28, 'bold'),
            text_color='#f0f6fc'
        )
        value_label.pack(anchor='w', pady=(5, 0))
        
        # Tendance
        if trend:
            color = '#00ff88' if '+' in trend else '#ff3860'
            trend_label = ctk.CTkLabel(
                container,
                text=trend,
                font=('Helvetica', 12),
                text_color=color
            )
            trend_label.pack(anchor='w', pady=(5, 0))


class GlassButton(ctk.CTkButton):
    """Bouton avec effet glassmorphism"""
    
    def __init__(self, parent, **kwargs):
        # Couleurs par défaut
        default_fg = kwargs.pop('fg_color', '#3b82f680')
        default_hover = kwargs.pop('hover_color', '#3b82f6a0')
        
        super().__init__(
            parent,
            fg_color=default_fg,
            hover_color=default_hover,
            corner_radius=12,
            border_width=1,
            border_color='#ffffff30',
            **kwargs
        )


class FloatingPanel(ctk.CTkFrame):
    """Panel flottant avec élévation"""
    
    def __init__(self, parent, elevation: int = 3, **kwargs):
        default_fg = kwargs.pop('fg_color', '#161b22')
        
        super().__init__(
            parent,
            fg_color=default_fg,
            corner_radius=15,
            border_width=1,
            border_color='#30363d',
            **kwargs
        )
        
        # Simuler l'élévation avec border
        if elevation > 2:
            self.configure(border_width=2, border_color='#3b82f640')


class GlassInput(ctk.CTkEntry):
    """Input avec effet glassmorphism"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color='#1e232980',
            border_width=1,
            border_color='#ffffff20',
            corner_radius=10,
            **kwargs
        )


class GlassScrollableFrame(ctk.CTkScrollableFrame):
    """Frame scrollable avec effet glassmorphism"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color='#1e232980',
            corner_radius=15,
            border_width=1,
            border_color='#ffffff20',
            **kwargs
        )


class GlassTabView(ctk.CTkTabview):
    """TabView avec effet glassmorphism"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color='#1e232980',
            corner_radius=15,
            border_width=1,
            border_color='#ffffff20',
            segmented_button_fg_color='#2a3038',
            segmented_button_selected_color='#3b82f6',
            **kwargs
        )


class GlassProgressBar(ctk.CTkProgressBar):
    """Barre de progression glassmorphism"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color='#1e232980',
            progress_color='#3b82f6',
            corner_radius=10,
            border_width=1,
            border_color='#ffffff20',
            **kwargs
        )


class GlassSwitch(ctk.CTkSwitch):
    """Switch avec effet glassmorphism"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color='#2a3038',
            progress_color='#3b82f6',
            button_color='#f0f6fc',
            button_hover_color='#ffffff',
            **kwargs
        )


class GlassSegmentedButton(ctk.CTkSegmentedButton):
    """Bouton segmenté glassmorphism"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color='#1e232980',
            selected_color='#3b82f6',
            selected_hover_color='#4a9eff',
            unselected_color='#2a3038',
            unselected_hover_color='#3a4048',
            corner_radius=10,
            border_width=1,
            **kwargs
        )


class NeumorphicFrame(ctk.CTkFrame):
    """Frame avec effet neumorphism"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color='#161b22',
            corner_radius=15,
            **kwargs
        )


class GlassModal(ctk.CTkToplevel):
    """Fenêtre modale avec effet glassmorphism"""
    
    def __init__(self, parent, title: str = "Modal", **kwargs):
        super().__init__(parent, **kwargs)
        
        self.title(title)
        self.geometry("500x400")
        
        # Rendre la fenêtre modale
        self.transient(parent)
        self.grab_set()
        
        # Frame principal
        self.main_frame = GlassFrame(self, border_glow=True)
        self.main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header = ctk.CTkFrame(self.main_frame, fg_color='transparent')
        header.pack(fill='x', padx=20, pady=20)
        
        ctk.CTkLabel(
            header,
            text=title,
            font=('Helvetica', 20, 'bold')
        ).pack(side='left')
        
        close_btn = ctk.CTkButton(
            header,
            text="✕",
            width=30,
            height=30,
            command=self.destroy,
            fg_color='#ff386040',
            hover_color='#ff3860'
        )
        close_btn.pack(side='right')
        
        # Conteneur pour le contenu
        self.content = ctk.CTkFrame(self.main_frame, fg_color='transparent')
        self.content.pack(fill='both', expand=True, padx=20, pady=(0, 20))


class GlassTooltip:
    """Tooltip avec effet glassmorphism"""
    
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tooltip = None
        
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)
    
    def show(self, event=None):
        """Affiche le tooltip"""
        if self.tooltip:
            return
        
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        
        self.tooltip = ctk.CTkToplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        frame = GlassFrame(self.tooltip)
        frame.pack(padx=2, pady=2)
        
        label = ctk.CTkLabel(
            frame,
            text=self.text,
            font=('Helvetica', 11)
        )
        label.pack(padx=10, pady=5)
    
    def hide(self, event=None):
        """Cache le tooltip"""
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None