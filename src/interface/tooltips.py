"""
HelixOne Tooltip System
Systeme de tooltips contextuels pour l'aide integree
"""

import customtkinter as ctk
from typing import Optional, Callable
import threading

from src.interface.design_system import COLORS, GLASS


class Tooltip:
    """Tooltip moderne qui apparait au survol d'un widget"""

    def __init__(
        self,
        widget: ctk.CTkBaseClass,
        text: str,
        delay: int = 500,  # Delai avant apparition (ms)
        position: str = "bottom",  # top, bottom, left, right
        max_width: int = 300
    ):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.position = position
        self.max_width = max_width

        self.tooltip_window: Optional[ctk.CTkToplevel] = None
        self._after_id: Optional[str] = None
        self._visible = False

        # Bindings
        self.widget.bind("<Enter>", self._on_enter)
        self.widget.bind("<Leave>", self._on_leave)
        self.widget.bind("<Button-1>", self._on_leave)

    def _on_enter(self, event=None):
        """Demarre le timer pour afficher le tooltip"""
        self._cancel_timer()
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, event=None):
        """Cache le tooltip"""
        self._cancel_timer()
        self._hide()

    def _cancel_timer(self):
        """Annule le timer"""
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        """Affiche le tooltip"""
        if self._visible:
            return

        self._visible = True

        # Creer la fenetre tooltip
        self.tooltip_window = ctk.CTkToplevel(self.widget)
        self.tooltip_window.withdraw()  # Cacher temporairement
        self.tooltip_window.overrideredirect(True)
        self.tooltip_window.configure(fg_color=GLASS['bg'])

        # Contenu
        frame = ctk.CTkFrame(
            self.tooltip_window,
            fg_color=GLASS['bg'],
            corner_radius=8,
            border_width=1,
            border_color=GLASS['border']
        )
        frame.pack(fill="both", expand=True)

        label = ctk.CTkLabel(
            frame,
            text=self.text,
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_primary'],
            wraplength=self.max_width,
            justify="left",
            padx=12,
            pady=8
        )
        label.pack()

        # Calculer la position
        self.tooltip_window.update_idletasks()
        self._position_tooltip()

        # Afficher avec un leger delai
        self.tooltip_window.deiconify()
        self.tooltip_window.lift()

    def _position_tooltip(self):
        """Positionne le tooltip selon la direction specifiee"""
        try:
            widget_x = self.widget.winfo_rootx()
            widget_y = self.widget.winfo_rooty()
            widget_w = self.widget.winfo_width()
            widget_h = self.widget.winfo_height()

            tip_w = self.tooltip_window.winfo_reqwidth()
            tip_h = self.tooltip_window.winfo_reqheight()

            screen_w = self.tooltip_window.winfo_screenwidth()
            screen_h = self.tooltip_window.winfo_screenheight()

            # Calculer position selon direction
            if self.position == "bottom":
                x = widget_x + (widget_w - tip_w) // 2
                y = widget_y + widget_h + 5
            elif self.position == "top":
                x = widget_x + (widget_w - tip_w) // 2
                y = widget_y - tip_h - 5
            elif self.position == "right":
                x = widget_x + widget_w + 5
                y = widget_y + (widget_h - tip_h) // 2
            else:  # left
                x = widget_x - tip_w - 5
                y = widget_y + (widget_h - tip_h) // 2

            # Garder dans l'ecran
            x = max(5, min(x, screen_w - tip_w - 5))
            y = max(5, min(y, screen_h - tip_h - 5))

            self.tooltip_window.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _hide(self):
        """Cache le tooltip"""
        self._visible = False
        if self.tooltip_window:
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass
            self.tooltip_window = None

    def update_text(self, new_text: str):
        """Met a jour le texte du tooltip"""
        self.text = new_text

    def destroy(self):
        """Nettoie le tooltip"""
        self._cancel_timer()
        self._hide()
        try:
            self.widget.unbind("<Enter>")
            self.widget.unbind("<Leave>")
            self.widget.unbind("<Button-1>")
        except Exception:
            pass


class HelpButton(ctk.CTkButton):
    """Bouton d'aide (?) avec tooltip integre"""

    def __init__(
        self,
        master,
        help_text: str,
        tooltip_position: str = "bottom",
        size: int = 24,
        **kwargs
    ):
        super().__init__(
            master,
            text="?",
            width=size,
            height=size,
            corner_radius=size // 2,
            fg_color=GLASS['bg'],
            hover_color=GLASS['bg_light'],
            text_color=COLORS['text_secondary'],
            font=ctk.CTkFont(size=size - 10, weight="bold"),
            **kwargs
        )

        self.help_text = help_text
        self.tooltip = Tooltip(
            self,
            text=help_text,
            position=tooltip_position,
            max_width=350
        )

    def destroy(self):
        """Nettoie le bouton et son tooltip"""
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()
        super().destroy()


class InfoLabel(ctk.CTkFrame):
    """Label avec icone d'information et tooltip"""

    def __init__(
        self,
        master,
        text: str,
        help_text: str,
        tooltip_position: str = "right",
        text_color: str = None,
        font: ctk.CTkFont = None,
        **kwargs
    ):
        super().__init__(master, fg_color="transparent", **kwargs)

        # Label principal
        self.label = ctk.CTkLabel(
            self,
            text=text,
            text_color=text_color or COLORS['text_primary'],
            font=font or ctk.CTkFont(size=14)
        )
        self.label.pack(side="left")

        # Icone info
        self.info_icon = ctk.CTkLabel(
            self,
            text="ℹ",
            text_color=COLORS['accent_cyan'],
            font=ctk.CTkFont(size=12),
            cursor="hand2"
        )
        self.info_icon.pack(side="left", padx=(5, 0))

        # Tooltip sur l'icone
        self.tooltip = Tooltip(
            self.info_icon,
            text=help_text,
            position=tooltip_position,
            delay=300
        )

    def destroy(self):
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()
        super().destroy()


# Dictionnaire des textes d'aide pour les differentes fonctionnalites
HELP_TEXTS = {
    # Dashboard
    "dashboard_overview": (
        "Le tableau de bord vous donne une vue d'ensemble de vos "
        "investissements, les tendances du marche et vos alertes actives."
    ),
    "portfolio_value": (
        "Valeur totale de votre portefeuille calculee en temps reel "
        "a partir des cours actuels du marche."
    ),
    "daily_change": (
        "Variation de la valeur de votre portefeuille depuis "
        "l'ouverture des marches aujourd'hui."
    ),

    # Analyseur
    "search_bar": (
        "Recherchez une action par son ticker (ex: AAPL) ou son nom "
        "(ex: Apple). Les suggestions apparaissent automatiquement."
    ),
    "ml_prediction": (
        "Prediction basee sur notre modele d'intelligence artificielle. "
        "Le score de confiance indique la fiabilite de la prediction."
    ),
    "technical_indicators": (
        "Indicateurs techniques calcules automatiquement: RSI, MACD, "
        "Moyennes mobiles, Bollinger Bands, etc."
    ),
    "sentiment_score": (
        "Score de sentiment agrege a partir des news, reseaux sociaux "
        "et analyses d'experts. De -100 (tres negatif) a +100 (tres positif)."
    ),

    # Alertes
    "price_alert": (
        "Definissez un seuil de prix. Vous serez notifie quand "
        "le cours atteint ou depasse ce niveau."
    ),
    "ml_signal_alert": (
        "Alertes automatiques basees sur les signaux de notre "
        "modele ML quand une opportunite est detectee."
    ),
    "volume_alert": (
        "Notification quand le volume d'echanges depasse un "
        "seuil inhabituel (potentiel mouvement de prix)."
    ),

    # Charts
    "chart_timeframe": (
        "Selectionnez la periode d'affichage: 1j, 1s, 1m, 3m, 1a, 5a. "
        "Les graphiques s'adaptent automatiquement."
    ),
    "chart_indicators": (
        "Ajoutez des indicateurs techniques au graphique: "
        "RSI, MACD, Bollinger Bands, etc."
    ),

    # Formation
    "formation_level": (
        "Votre niveau actuel dans la Formation HelixOne. "
        "Completez les modules pour progresser et debloquer du contenu."
    ),
    "badges": (
        "Badges gagnes en completant des modules et des quiz. "
        "Chaque badge temoigne d'une competence acquise."
    ),

    # Crypto
    "crypto_portfolio": (
        "Vue d'ensemble de vos positions en cryptomonnaies. "
        "Les prix sont mis a jour en temps reel."
    ),

    # IBKR
    "ibkr_connection": (
        "Connectez votre compte Interactive Brokers pour "
        "synchroniser votre portefeuille et passer des ordres."
    ),

    # Raccourcis
    "keyboard_shortcuts": (
        "Appuyez sur F1 a tout moment pour voir la liste "
        "complete des raccourcis clavier disponibles."
    ),
}


def add_tooltip(widget: ctk.CTkBaseClass, text: str, **kwargs) -> Tooltip:
    """
    Ajoute un tooltip a un widget existant.

    Usage:
        button = ctk.CTkButton(parent, text="Click")
        add_tooltip(button, "Cliquez pour confirmer")
    """
    return Tooltip(widget, text, **kwargs)


def get_help_text(key: str) -> str:
    """Recupere un texte d'aide par sa cle"""
    return HELP_TEXTS.get(key, "Aide non disponible")
