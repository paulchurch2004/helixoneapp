"""
Composants UI améliorés pour HelixOne
- Animations fluides
- États de chargement
- Effets visuels
- Composants réutilisables
"""

import customtkinter as ctk
from typing import Callable, Optional, Tuple, List
import threading
import time


# =============================================================================
# CONSTANTES D'ANIMATION
# =============================================================================
DURATION_FAST = 150      # ms
DURATION_NORMAL = 250    # ms
DURATION_SLOW = 400      # ms
ANIMATION_FPS = 60


# =============================================================================
# BOUTON ANIMÉ AVEC EFFETS HOVER
# =============================================================================
class AnimatedButton(ctk.CTkButton):
    """
    Bouton avec animations au survol:
    - Changement de couleur fluide
    - Léger scale effect
    - Feedback visuel au clic
    """

    def __init__(self, master, hover_scale: float = 1.02, **kwargs):
        self.hover_scale = hover_scale
        self.original_width = kwargs.get('width', 100)
        self.original_height = kwargs.get('height', 32)
        self._animating = False

        super().__init__(master, **kwargs)

        # Bind events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, event=None):
        """Animation au survol"""
        if self._animating:
            return
        try:
            new_width = int(self.original_width * self.hover_scale)
            new_height = int(self.original_height * self.hover_scale)
            self.configure(width=new_width, height=new_height)
        except Exception:
            pass

    def _on_leave(self, event=None):
        """Retour à la taille normale"""
        try:
            self.configure(width=self.original_width, height=self.original_height)
        except Exception:
            pass

    def _on_click(self, event=None):
        """Effet au clic"""
        try:
            self.configure(
                width=int(self.original_width * 0.98),
                height=int(self.original_height * 0.98)
            )
        except Exception:
            pass

    def _on_release(self, event=None):
        """Retour après clic"""
        self.after(50, lambda: self._on_enter(None))


# =============================================================================
# SPINNER DE CHARGEMENT
# =============================================================================
class LoadingSpinner(ctk.CTkFrame):
    """
    Spinner de chargement animé avec points qui pulsent
    """

    def __init__(self, master, size: int = 40, color: str = "#00D9FF", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.size = size
        self.color = color
        self.dots = []
        self.running = False
        self.current_dot = 0

        self._create_dots()

    def _create_dots(self):
        """Créer les 3 points du spinner"""
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(expand=True)

        dot_size = self.size // 4

        for i in range(3):
            dot = ctk.CTkFrame(
                container,
                width=dot_size,
                height=dot_size,
                corner_radius=dot_size // 2,
                fg_color=self.color
            )
            dot.pack(side="left", padx=3)
            self.dots.append(dot)

    def start(self):
        """Démarrer l'animation"""
        self.running = True
        self._animate()

    def stop(self):
        """Arrêter l'animation"""
        self.running = False
        # Reset opacity
        for dot in self.dots:
            try:
                dot.configure(fg_color=self.color)
            except Exception:
                pass

    def _animate(self):
        """Animation des points"""
        if not self.running:
            return

        try:
            # Réduire l'opacité de tous les points
            dim_color = self._dim_color(self.color, 0.3)
            for dot in self.dots:
                dot.configure(fg_color=dim_color)

            # Mettre en évidence le point actuel
            self.dots[self.current_dot].configure(fg_color=self.color)

            self.current_dot = (self.current_dot + 1) % 3

            self.after(200, self._animate)
        except Exception:
            pass

    def _dim_color(self, hex_color: str, factor: float) -> str:
        """Assombrir une couleur"""
        hex_color = hex_color.lstrip('#')
        r = int(int(hex_color[0:2], 16) * factor)
        g = int(int(hex_color[2:4], 16) * factor)
        b = int(int(hex_color[4:6], 16) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"


# =============================================================================
# SKELETON LOADER
# =============================================================================
class SkeletonLoader(ctk.CTkFrame):
    """
    Placeholder animé pendant le chargement des données
    Effet shimmer/pulse
    """

    def __init__(self, master, width: int = 200, height: int = 20,
                 corner_radius: int = 4, **kwargs):
        super().__init__(
            master,
            width=width,
            height=height,
            corner_radius=corner_radius,
            fg_color="#2a2d36",
            **kwargs
        )
        self.pack_propagate(False)

        self.base_color = "#2a2d36"
        self.highlight_color = "#3a3d46"
        self.running = False

    def start(self):
        """Démarrer l'animation shimmer"""
        self.running = True
        self._shimmer()

    def stop(self):
        """Arrêter l'animation"""
        self.running = False
        try:
            self.configure(fg_color=self.base_color)
        except Exception:
            pass

    def _shimmer(self):
        """Animation shimmer"""
        if not self.running:
            return

        try:
            # Alterner entre les couleurs
            current = self.cget("fg_color")
            if current == self.base_color:
                self.configure(fg_color=self.highlight_color)
            else:
                self.configure(fg_color=self.base_color)

            self.after(400, self._shimmer)
        except Exception:
            pass


# =============================================================================
# CARD SKELETON (pour les cartes de données)
# =============================================================================
class CardSkeleton(ctk.CTkFrame):
    """
    Skeleton pour une carte de données complète
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#1a1d24", corner_radius=10, **kwargs)

        self.skeletons = []
        self._create_skeleton()

    def _create_skeleton(self):
        """Créer le layout skeleton"""
        # Titre
        title = SkeletonLoader(self, width=150, height=20)
        title.pack(padx=15, pady=(15, 10), anchor="w")
        self.skeletons.append(title)

        # Lignes de contenu
        for _ in range(3):
            line = SkeletonLoader(self, width=250, height=14)
            line.pack(padx=15, pady=5, anchor="w")
            self.skeletons.append(line)

        # Valeur principale
        value = SkeletonLoader(self, width=100, height=30)
        value.pack(padx=15, pady=(10, 15), anchor="w")
        self.skeletons.append(value)

    def start(self):
        """Démarrer toutes les animations"""
        for skeleton in self.skeletons:
            skeleton.start()

    def stop(self):
        """Arrêter toutes les animations"""
        for skeleton in self.skeletons:
            skeleton.stop()


# =============================================================================
# INDICATEUR DE TENDANCE
# =============================================================================
class TrendIndicator(ctk.CTkFrame):
    """
    Indicateur de tendance avec flèche et couleur
    """

    def __init__(self, master, value: float = 0, show_value: bool = True,
                 size: str = "normal", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.value = value
        self.show_value = show_value
        self.size = size

        self._create_indicator()

    def _create_indicator(self):
        """Créer l'indicateur"""
        # Déterminer la direction et la couleur
        if self.value > 0:
            arrow = "▲"
            color = "#00FF88"
        elif self.value < 0:
            arrow = "▼"
            color = "#FF4444"
        else:
            arrow = "●"
            color = "#888888"

        # Taille du texte
        font_size = 14 if self.size == "normal" else 11 if self.size == "small" else 18

        # Container
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack()

        # Flèche
        arrow_label = ctk.CTkLabel(
            container,
            text=arrow,
            font=ctk.CTkFont(size=font_size, weight="bold"),
            text_color=color
        )
        arrow_label.pack(side="left", padx=(0, 3))

        # Valeur
        if self.show_value:
            value_text = f"{abs(self.value):+.2f}%" if self.value != 0 else "0.00%"
            value_label = ctk.CTkLabel(
                container,
                text=value_text,
                font=ctk.CTkFont(size=font_size - 2),
                text_color=color
            )
            value_label.pack(side="left")

    def update_value(self, new_value: float):
        """Mettre à jour la valeur"""
        self.value = new_value
        # Recréer l'indicateur
        for widget in self.winfo_children():
            widget.destroy()
        self._create_indicator()


# =============================================================================
# TOAST NOTIFICATION AMÉLIORÉ
# =============================================================================
class EnhancedToast(ctk.CTkFrame):
    """
    Toast notification avec:
    - Animation d'entrée/sortie
    - Barre de progression
    - Actions (boutons)
    """

    def __init__(self, master, message: str, severity: str = "info",
                 duration: int = 4000, action_text: str = None,
                 action_callback: Callable = None, **kwargs):

        # Couleurs par sévérité
        self.colors = {
            "info": ("#1a4a6e", "#00D9FF"),
            "success": ("#1a4e3a", "#00FF88"),
            "warning": ("#4e4a1a", "#FFD700"),
            "error": ("#4e1a1a", "#FF4444"),
            "critical": ("#6e1a1a", "#FF0000")
        }

        bg_color, accent_color = self.colors.get(severity, self.colors["info"])

        super().__init__(master, fg_color=bg_color, corner_radius=10, **kwargs)

        self.message = message
        self.severity = severity
        self.duration = duration
        self.accent_color = accent_color
        self.action_callback = action_callback
        self.progress = 0

        self._create_ui(action_text)
        self._animate_in()

    def _create_ui(self, action_text: str):
        """Créer l'interface du toast"""
        # Container principal
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=12)

        # Icône
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨"
        }

        icon_label = ctk.CTkLabel(
            main,
            text=icons.get(self.severity, "ℹ️"),
            font=ctk.CTkFont(size=18)
        )
        icon_label.pack(side="left", padx=(0, 10))

        # Message
        msg_label = ctk.CTkLabel(
            main,
            text=self.message,
            font=ctk.CTkFont(size=13),
            text_color="#FFFFFF",
            wraplength=300
        )
        msg_label.pack(side="left", fill="x", expand=True)

        # Bouton action
        if action_text and self.action_callback:
            action_btn = ctk.CTkButton(
                main,
                text=action_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=self.accent_color,
                hover_color=self._lighten_color(self.accent_color),
                text_color="#000000",
                width=70, height=28,
                corner_radius=6,
                command=self._on_action
            )
            action_btn.pack(side="left", padx=(10, 0))

        # Bouton fermer
        close_btn = ctk.CTkButton(
            main,
            text="✕",
            width=28, height=28,
            fg_color="transparent",
            hover_color="#ffffff20",
            text_color="#AAAAAA",
            command=self._close
        )
        close_btn.pack(side="right", padx=(10, 0))

        # Barre de progression
        self.progress_bar = ctk.CTkFrame(
            self,
            height=3,
            corner_radius=0,
            fg_color=self.accent_color
        )
        self.progress_bar.place(x=0, y=0, relwidth=1)

        # Démarrer la progression
        if self.duration > 0:
            self._update_progress()

    def _animate_in(self):
        """Animation d'entrée (slide up + fade)"""
        # Positionner en bas et animer vers le haut
        self.place(relx=1, rely=1, anchor="se", x=-20, y=20)  # Start off-screen
        self._slide_in(0)

    def _slide_in(self, step: int):
        """Animation slide"""
        if step <= 10:
            y_offset = 20 - (step * 4)  # 20 -> -20
            self.place(relx=1, rely=1, anchor="se", x=-20, y=y_offset)
            self.after(20, lambda: self._slide_in(step + 1))

    def _update_progress(self):
        """Mettre à jour la barre de progression"""
        if self.progress >= 100:
            self._close()
            return

        try:
            self.progress += 100 / (self.duration / 50)
            width = 1 - (self.progress / 100)
            self.progress_bar.place(x=0, y=0, relwidth=width)
            self.after(50, self._update_progress)
        except Exception:
            pass

    def _on_action(self):
        """Action button clicked"""
        if self.action_callback:
            self.action_callback()
        self._close()

    def _close(self):
        """Fermer le toast avec animation"""
        self._slide_out(0)

    def _slide_out(self, step: int):
        """Animation de sortie"""
        if step <= 10:
            y_offset = -20 + (step * 6)
            try:
                self.place(relx=1, rely=1, anchor="se", x=-20, y=y_offset)
                self.after(15, lambda: self._slide_out(step + 1))
            except Exception:
                pass
        else:
            try:
                self.destroy()
            except Exception:
                pass

    def _lighten_color(self, hex_color: str) -> str:
        """Éclaircir une couleur"""
        hex_color = hex_color.lstrip('#')
        r = min(255, int(int(hex_color[0:2], 16) * 1.2))
        g = min(255, int(int(hex_color[2:4], 16) * 1.2))
        b = min(255, int(int(hex_color[4:6], 16) * 1.2))
        return f"#{r:02x}{g:02x}{b:02x}"


# =============================================================================
# ÉTAT VIDE
# =============================================================================
class EmptyState(ctk.CTkFrame):
    """
    Affichage d'état vide avec illustration et CTA
    """

    def __init__(self, master, title: str, description: str = "",
                 icon: str = "📭", action_text: str = None,
                 action_callback: Callable = None, **kwargs):

        super().__init__(master, fg_color="transparent", **kwargs)

        self._create_ui(title, description, icon, action_text, action_callback)

    def _create_ui(self, title, description, icon, action_text, action_callback):
        """Créer l'interface"""
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(expand=True)

        # Icône
        icon_label = ctk.CTkLabel(
            container,
            text=icon,
            font=ctk.CTkFont(size=64)
        )
        icon_label.pack(pady=(0, 20))

        # Titre
        title_label = ctk.CTkLabel(
            container,
            text=title,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#FFFFFF"
        )
        title_label.pack(pady=(0, 10))

        # Description
        if description:
            desc_label = ctk.CTkLabel(
                container,
                text=description,
                font=ctk.CTkFont(size=13),
                text_color="#888888",
                wraplength=300
            )
            desc_label.pack(pady=(0, 20))

        # Bouton action
        if action_text and action_callback:
            action_btn = ctk.CTkButton(
                container,
                text=action_text,
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color="#00D9FF",
                hover_color="#00B8E6",
                text_color="#000000",
                height=40,
                corner_radius=8,
                command=action_callback
            )
            action_btn.pack()


# =============================================================================
# VALIDATION INLINE
# =============================================================================
class ValidatedEntry(ctk.CTkFrame):
    """
    Champ de saisie avec validation en temps réel
    """

    def __init__(self, master, label: str = "", placeholder: str = "",
                 validator: Callable = None, error_message: str = "",
                 show: str = None, **kwargs):

        super().__init__(master, fg_color="transparent", **kwargs)

        self.validator = validator
        self.error_message = error_message
        self.is_valid = True

        self._create_ui(label, placeholder, show)

    def _create_ui(self, label, placeholder, show):
        """Créer l'interface"""
        # Label
        if label:
            label_frame = ctk.CTkFrame(self, fg_color="transparent")
            label_frame.pack(fill="x")

            self.label = ctk.CTkLabel(
                label_frame,
                text=label,
                font=ctk.CTkFont(size=12),
                text_color="#AAAAAA"
            )
            self.label.pack(side="left")

            # Indicateur de validation
            self.status_label = ctk.CTkLabel(
                label_frame,
                text="",
                font=ctk.CTkFont(size=12)
            )
            self.status_label.pack(side="right")

        # Champ de saisie
        entry_kwargs = {
            "height": 40,
            "font": ctk.CTkFont(size=13),
            "fg_color": "#1a1d24",
            "border_color": "#2a2d36",
            "placeholder_text": placeholder
        }
        if show:
            entry_kwargs["show"] = show

        self.entry = ctk.CTkEntry(self, **entry_kwargs)
        self.entry.pack(fill="x", pady=(5, 0))

        # Message d'erreur
        self.error_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#FF4444",
            anchor="w"
        )
        self.error_label.pack(fill="x", pady=(3, 0))

        # Bind validation
        self.entry.bind("<KeyRelease>", self._validate)
        self.entry.bind("<FocusOut>", self._validate)

    def _validate(self, event=None):
        """Valider le contenu"""
        if not self.validator:
            return True

        value = self.entry.get()
        if not value:
            self._set_neutral()
            return True

        if self.validator(value):
            self._set_valid()
            return True
        else:
            self._set_invalid()
            return False

    def _set_valid(self):
        """Marquer comme valide"""
        self.is_valid = True
        self.entry.configure(border_color="#00FF88")
        if hasattr(self, 'status_label'):
            self.status_label.configure(text="✓", text_color="#00FF88")
        self.error_label.configure(text="")

    def _set_invalid(self):
        """Marquer comme invalide"""
        self.is_valid = False
        self.entry.configure(border_color="#FF4444")
        if hasattr(self, 'status_label'):
            self.status_label.configure(text="✗", text_color="#FF4444")
        self.error_label.configure(text=self.error_message)

    def _set_neutral(self):
        """État neutre"""
        self.is_valid = True
        self.entry.configure(border_color="#2a2d36")
        if hasattr(self, 'status_label'):
            self.status_label.configure(text="")
        self.error_label.configure(text="")

    def get(self) -> str:
        """Obtenir la valeur"""
        return self.entry.get()

    def validate(self) -> bool:
        """Forcer la validation"""
        return self._validate()


# =============================================================================
# PASSWORD STRENGTH METER
# =============================================================================
class PasswordStrengthMeter(ctk.CTkFrame):
    """
    Indicateur de force de mot de passe
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.strength = 0
        self._create_ui()

    def _create_ui(self):
        """Créer l'interface"""
        # Barres de force
        self.bars_frame = ctk.CTkFrame(self, fg_color="transparent", height=4)
        self.bars_frame.pack(fill="x", pady=(5, 3))

        self.bars = []
        for i in range(4):
            bar = ctk.CTkFrame(
                self.bars_frame,
                height=4,
                corner_radius=2,
                fg_color="#2a2d36"
            )
            bar.pack(side="left", fill="x", expand=True, padx=1)
            self.bars.append(bar)

        # Label de force
        self.strength_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.strength_label.pack(anchor="w")

    def update_strength(self, password: str):
        """Mettre à jour la force"""
        self.strength = self._calculate_strength(password)
        self._update_ui()

    def _calculate_strength(self, password: str) -> int:
        """Calculer la force du mot de passe (0-4)"""
        if not password:
            return 0

        score = 0

        # Longueur
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1

        # Complexité
        if any(c.isupper() for c in password):
            score += 0.5
        if any(c.islower() for c in password):
            score += 0.5
        if any(c.isdigit() for c in password):
            score += 0.5
        if any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in password):
            score += 0.5

        return min(4, int(score))

    def _update_ui(self):
        """Mettre à jour l'affichage"""
        colors = ["#FF4444", "#FF8844", "#FFBB44", "#00FF88"]
        labels = ["Très faible", "Faible", "Moyen", "Fort"]

        for i, bar in enumerate(self.bars):
            if i < self.strength:
                bar.configure(fg_color=colors[min(self.strength - 1, 3)])
            else:
                bar.configure(fg_color="#2a2d36")

        if self.strength > 0:
            self.strength_label.configure(
                text=labels[self.strength - 1],
                text_color=colors[self.strength - 1]
            )
        else:
            self.strength_label.configure(text="", text_color="#888888")


# =============================================================================
# ANIMATED COUNTER
# =============================================================================
class AnimatedCounter(ctk.CTkLabel):
    """
    Compteur animé (1000 -> 1234 avec animation)
    """

    def __init__(self, master, start_value: float = 0, prefix: str = "",
                 suffix: str = "", decimals: int = 0, duration: int = 500,
                 **kwargs):

        self.current_value = start_value
        self.target_value = start_value
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.duration = duration

        super().__init__(master, text=self._format_value(start_value), **kwargs)

    def _format_value(self, value: float) -> str:
        """Formater la valeur"""
        if self.decimals == 0:
            formatted = f"{int(value):,}"
        else:
            formatted = f"{value:,.{self.decimals}f}"
        return f"{self.prefix}{formatted}{self.suffix}"

    def set_value(self, new_value: float, animate: bool = True):
        """Définir une nouvelle valeur avec animation"""
        if not animate:
            self.current_value = new_value
            self.configure(text=self._format_value(new_value))
            return

        self.target_value = new_value
        steps = 20
        step_duration = self.duration // steps
        diff = (new_value - self.current_value) / steps

        self._animate_step(diff, steps, step_duration)

    def _animate_step(self, diff: float, remaining_steps: int, step_duration: int):
        """Étape d'animation"""
        if remaining_steps <= 0:
            self.current_value = self.target_value
            self.configure(text=self._format_value(self.current_value))
            return

        self.current_value += diff
        self.configure(text=self._format_value(self.current_value))
        self.after(step_duration, lambda: self._animate_step(diff, remaining_steps - 1, step_duration))


# =============================================================================
# PANEL TRANSITION WRAPPER
# =============================================================================
class TransitionPanel(ctk.CTkFrame):
    """
    Container avec transitions animées entre les panels
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.current_panel = None
        self._transitioning = False

    def show_panel(self, panel_class, *args, transition: str = "fade", **kwargs):
        """
        Afficher un nouveau panel avec transition

        Args:
            panel_class: Classe du panel à afficher
            transition: "fade", "slide_left", "slide_right", "none"
        """
        if self._transitioning:
            return

        new_panel = panel_class(self, *args, **kwargs)

        if self.current_panel is None or transition == "none":
            # Pas de transition
            new_panel.pack(fill="both", expand=True)
            self.current_panel = new_panel
            return

        self._transitioning = True

        if transition == "fade":
            self._fade_transition(new_panel)
        elif transition == "slide_left":
            self._slide_transition(new_panel, "left")
        elif transition == "slide_right":
            self._slide_transition(new_panel, "right")
        else:
            new_panel.pack(fill="both", expand=True)
            if self.current_panel:
                self.current_panel.destroy()
            self.current_panel = new_panel
            self._transitioning = False

    def _fade_transition(self, new_panel):
        """Transition en fondu"""
        # Pour CTk, on simule le fade avec place/pack
        new_panel.place(relx=0, rely=0, relwidth=1, relheight=1)

        def complete():
            if self.current_panel:
                self.current_panel.destroy()
            new_panel.place_forget()
            new_panel.pack(fill="both", expand=True)
            self.current_panel = new_panel
            self._transitioning = False

        self.after(DURATION_NORMAL, complete)

    def _slide_transition(self, new_panel, direction: str):
        """Transition en glissement"""
        start_x = 1 if direction == "left" else -1

        new_panel.place(relx=start_x, rely=0, relwidth=1, relheight=1)

        steps = 10
        step_size = start_x / steps

        def animate(current_step):
            if current_step >= steps:
                if self.current_panel:
                    self.current_panel.destroy()
                new_panel.place_forget()
                new_panel.pack(fill="both", expand=True)
                self.current_panel = new_panel
                self._transitioning = False
                return

            x = start_x - (step_size * (current_step + 1))
            new_panel.place(relx=x, rely=0, relwidth=1, relheight=1)
            self.after(DURATION_NORMAL // steps, lambda: animate(current_step + 1))

        animate(0)


# =============================================================================
# ETAT VIDE - Composant pour ecrans sans donnees
# =============================================================================
class EmptyState(ctk.CTkFrame):
    """
    Composant affiche quand il n'y a pas de donnees.
    Affiche un message encourageant et un bouton d'action.
    """

    # Presets d'etats vides
    PRESETS = {
        "portfolio": {
            "icon": "📊",
            "title": "Votre portfolio est vide",
            "message": "Commencez par ajouter des actions a votre portfolio pour suivre vos investissements.",
            "action_text": "Rechercher une action",
            "action_icon": "🔍"
        },
        "alerts": {
            "icon": "🔔",
            "title": "Aucune alerte configuree",
            "message": "Creez des alertes pour etre notifie des opportunites de marche.",
            "action_text": "Creer une alerte",
            "action_icon": "+"
        },
        "watchlist": {
            "icon": "👀",
            "title": "Votre watchlist est vide",
            "message": "Ajoutez des actions pour les suivre facilement.",
            "action_text": "Ajouter une action",
            "action_icon": "+"
        },
        "search": {
            "icon": "🔍",
            "title": "Aucun resultat",
            "message": "Essayez avec un autre terme de recherche.",
            "action_text": None,
            "action_icon": None
        },
        "crypto": {
            "icon": "₿",
            "title": "Aucune crypto suivie",
            "message": "Ajoutez des cryptomonnaies pour suivre leurs cours en temps reel.",
            "action_text": "Explorer les cryptos",
            "action_icon": "🚀"
        },
        "formation": {
            "icon": "🎓",
            "title": "Bienvenue dans l'Academy",
            "message": "Commencez votre parcours de formation pour maitriser le trading.",
            "action_text": "Commencer",
            "action_icon": "▶"
        },
        "error": {
            "icon": "⚠️",
            "title": "Erreur de chargement",
            "message": "Impossible de charger les donnees. Verifiez votre connexion.",
            "action_text": "Reessayer",
            "action_icon": "🔄"
        },
        "loading": {
            "icon": "⏳",
            "title": "Chargement en cours",
            "message": "Veuillez patienter...",
            "action_text": None,
            "action_icon": None
        }
    }

    def __init__(
        self,
        master,
        preset: str = None,
        icon: str = "📭",
        title: str = "Aucune donnee",
        message: str = "Il n'y a rien a afficher pour le moment.",
        action_text: str = None,
        action_callback: Callable = None,
        action_icon: str = None,
        **kwargs
    ):
        # Couleurs
        bg_color = kwargs.pop('fg_color', '#1a1d24')
        super().__init__(master, fg_color=bg_color, **kwargs)

        # Utiliser preset si fourni
        if preset and preset in self.PRESETS:
            p = self.PRESETS[preset]
            icon = p["icon"]
            title = p["title"]
            message = p["message"]
            action_text = action_text or p["action_text"]
            action_icon = action_icon or p["action_icon"]

        self.action_callback = action_callback
        self._create_ui(icon, title, message, action_text, action_icon)

    def _create_ui(self, icon: str, title: str, message: str, action_text: str, action_icon: str):
        """Cree l'interface de l'etat vide"""
        # Container centre
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.place(relx=0.5, rely=0.5, anchor="center")

        # Grande icone
        icon_label = ctk.CTkLabel(
            container,
            text=icon,
            font=ctk.CTkFont(size=64)
        )
        icon_label.pack(pady=(0, 20))

        # Titre
        title_label = ctk.CTkLabel(
            container,
            text=title,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#FFFFFF"
        )
        title_label.pack(pady=(0, 10))

        # Message
        message_label = ctk.CTkLabel(
            container,
            text=message,
            font=ctk.CTkFont(size=14),
            text_color="#888888",
            wraplength=350,
            justify="center"
        )
        message_label.pack(pady=(0, 25))

        # Bouton action
        if action_text and self.action_callback:
            btn_text = f"{action_icon}  {action_text}" if action_icon else action_text
            action_btn = ctk.CTkButton(
                container,
                text=btn_text,
                font=ctk.CTkFont(size=14, weight="bold"),
                fg_color="#00D4FF",
                hover_color="#00A8CC",
                text_color="#000000",
                corner_radius=10,
                width=200,
                height=45,
                command=self.action_callback
            )
            action_btn.pack()


# =============================================================================
# NOTIFICATION DE BIENVENUE
# =============================================================================
class WelcomeNotification(ctk.CTkToplevel):
    """Notification de bienvenue avec animation"""

    def __init__(self, master, user_name: str = ""):
        super().__init__(master)

        # Configuration
        self.overrideredirect(True)
        self.configure(fg_color="#1e2329")
        self.attributes('-alpha', 0.0)

        # Taille et position
        width, height = 350, 100
        screen_w = self.winfo_screenwidth()
        x = screen_w - width - 30
        y = 30
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Contenu
        frame = ctk.CTkFrame(
            self,
            fg_color="#1e2329",
            corner_radius=12,
            border_width=1,
            border_color="#00D4FF"
        )
        frame.pack(fill="both", expand=True, padx=2, pady=2)

        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=15, pady=12)

        # Emoji et texte
        emoji = ctk.CTkLabel(inner, text="👋", font=ctk.CTkFont(size=32))
        emoji.pack(side="left", padx=(0, 12))

        text_frame = ctk.CTkFrame(inner, fg_color="transparent")
        text_frame.pack(side="left", fill="both", expand=True)

        greeting = f"Bienvenue{', ' + user_name if user_name else ''} !"
        title = ctk.CTkLabel(
            text_frame,
            text=greeting,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FFFFFF",
            anchor="w"
        )
        title.pack(fill="x")

        subtitle = ctk.CTkLabel(
            text_frame,
            text="Pret a analyser les marches ?",
            font=ctk.CTkFont(size=12),
            text_color="#888888",
            anchor="w"
        )
        subtitle.pack(fill="x")

        # Bouton fermer
        close_btn = ctk.CTkButton(
            inner,
            text="✕",
            width=28, height=28,
            fg_color="transparent",
            hover_color="#ffffff20",
            text_color="#888888",
            command=self._close
        )
        close_btn.pack(side="right")

        # Animations
        self._fade_in()
        self.after(5000, self._fade_out)  # Auto-fermeture apres 5s

    def _fade_in(self):
        """Animation fade in"""
        alpha = 0.0
        def animate():
            nonlocal alpha
            alpha += 0.1
            if alpha <= 1.0:
                self.attributes('-alpha', alpha)
                self.after(30, animate)
        animate()

    def _fade_out(self):
        """Animation fade out"""
        alpha = 1.0
        def animate():
            nonlocal alpha
            alpha -= 0.1
            if alpha >= 0:
                self.attributes('-alpha', alpha)
                self.after(30, animate)
            else:
                self.destroy()
        animate()

    def _close(self):
        """Ferme la notification"""
        self._fade_out()
