"""
HelixOne Design System - Dark Glassmorphism

Systeme de design centralise pour une interface coherente et professionnelle.
Tous les composants doivent utiliser ces tokens au lieu de couleurs hardcodees.
"""

from typing import Dict, Any

# =============================================================================
# DESIGN TOKENS - Source unique de verite pour le design
# =============================================================================

DESIGN_TOKENS: Dict[str, Any] = {
    # --- Effets Glass (Glassmorphism) ---
    # Note: CustomTkinter ne supporte pas l'alpha, on simule avec des couleurs sombres
    'glass': {
        'bg': '#1e2329',                # Fond glass (simule transparence)
        'bg_light': '#2a3038',          # Fond glass plus clair
        'bg_dark': '#252a33',           # Fond glass pour inputs (plus visible)
        'border': '#4a5568',            # Bordure plus visible
        'border_hover': '#5a6578',      # Bordure au survol
        'border_active': '#00D4FF',     # Bordure cyan quand actif
        'glow': '#00D4FF',              # Lueur cyan
        'glow_green': '#00ff88',        # Lueur verte
        'radius_sm': 8,                 # Petit rayon (boutons)
        'radius': 12,                   # Rayon standard
        'radius_lg': 16,                # Grand rayon (panels)
        'radius_xl': 20,                # Tres grand rayon (modals)
    },

    # --- Palette de Couleurs ---
    'colors': {
        # Fonds
        'bg_primary': '#0d1117',         # Fond principal (tres sombre)
        'bg_secondary': '#161b22',       # Fond secondaire
        'bg_tertiary': '#1e2329',        # Fond tertiaire
        'bg_elevated': '#21262d',        # Fond eleve (cards)
        'bg_hover': '#2a303880',         # Fond au survol

        # Accents
        'accent_cyan': '#00D4FF',        # Accent principal (cyan)
        'accent_cyan_dim': '#00D4FF80',  # Cyan attenue
        'accent_green': '#00ff88',       # Accent vert (succes/positif)
        'accent_green_dim': '#00ff8880', # Vert attenue
        'accent_blue': '#3b82f6',        # Bleu complementaire
        'accent_purple': '#8b5cf6',      # Violet pour charts

        # Texte
        'text_primary': '#f0f6fc',       # Texte principal (blanc casse)
        'text_secondary': '#8b949e',     # Texte secondaire (gris)
        'text_muted': '#64748b',         # Texte attenue
        'text_disabled': '#484f58',      # Texte desactive

        # Bordures
        'border': '#30363d',             # Bordure standard
        'border_light': '#3d444d',       # Bordure plus claire
        'border_focus': '#00D4FF',       # Bordure focus

        # Statuts
        'success': '#00ff88',            # Succes (vert)
        'warning': '#ffaa00',            # Avertissement (orange)
        'error': '#ff3860',              # Erreur (rouge)
        'info': '#00D4FF',               # Info (cyan)

        # Trading
        'bullish': '#00ff88',            # Haussier (vert)
        'bearish': '#ff3860',            # Baissier (rouge)
        'neutral': '#8b949e',            # Neutre (gris)
    },

    # --- Ombres et Effets ---
    'effects': {
        'shadow_sm': '0 2px 4px rgba(0,0,0,0.3)',
        'shadow': '0 4px 12px rgba(0,0,0,0.4)',
        'shadow_lg': '0 8px 24px rgba(0,0,0,0.5)',
        'shadow_glow_cyan': '0 0 20px rgba(0,212,255,0.3)',
        'shadow_glow_green': '0 0 20px rgba(0,255,136,0.3)',
        'backdrop_blur': 12,  # px
    },

    # --- Typographie ---
    'typography': {
        'font_family': ('Segoe UI', 'Roboto', 'Helvetica', 'Arial'),
        'font_mono': ('Consolas', 'Monaco', 'Courier New'),

        # Tailles
        'size_xs': 11,
        'size_sm': 12,
        'size_base': 14,
        'size_lg': 16,
        'size_xl': 18,
        'size_2xl': 22,
        'size_3xl': 28,
        'size_4xl': 36,

        # Poids
        'weight_normal': 'normal',
        'weight_medium': 'bold',  # CTk n'a pas medium
        'weight_bold': 'bold',
    },

    # --- Espacements ---
    'spacing': {
        'xs': 4,
        'sm': 8,
        'md': 12,
        'lg': 16,
        'xl': 24,
        '2xl': 32,
        '3xl': 48,
        '4xl': 64,
    },

    # --- Animations ---
    'animation': {
        'fast': 150,       # ms - hover, micro-interactions
        'normal': 250,     # ms - transitions standard
        'slow': 400,       # ms - transitions complexes
        'very_slow': 600,  # ms - animations dramatiques
    },

    # --- Dimensions ---
    'dimensions': {
        'sidebar_expanded': 240,
        'sidebar_collapsed': 60,
        'header_height': 60,
        'button_height': 40,
        'input_height': 45,
        'card_min_width': 280,
    },
}


# =============================================================================
# FONCTIONS HELPER - Acces facile aux tokens
# =============================================================================

def get_color(key: str, fallback: str = '#ffffff') -> str:
    """Recupere une couleur du design system."""
    return DESIGN_TOKENS['colors'].get(key, fallback)


def get_glass(key: str) -> Any:
    """Recupere une valeur glass du design system."""
    return DESIGN_TOKENS['glass'].get(key)


def get_spacing(key: str) -> int:
    """Recupere un espacement du design system."""
    return DESIGN_TOKENS['spacing'].get(key, 16)


def get_animation(key: str) -> int:
    """Recupere une duree d'animation en ms."""
    return DESIGN_TOKENS['animation'].get(key, 250)


def get_font_size(key: str) -> int:
    """Recupere une taille de police."""
    return DESIGN_TOKENS['typography'].get(key, 14)


# =============================================================================
# STYLES PREDEFINIS - Configurations pret-a-l'emploi
# =============================================================================

GLASS_PANEL_STYLE = {
    'fg_color': DESIGN_TOKENS['glass']['bg'],
    'corner_radius': DESIGN_TOKENS['glass']['radius_lg'],
    'border_width': 1,
    'border_color': DESIGN_TOKENS['glass']['border'],
}

GLASS_CARD_STYLE = {
    'fg_color': DESIGN_TOKENS['glass']['bg_light'],
    'corner_radius': DESIGN_TOKENS['glass']['radius'],
    'border_width': 1,
    'border_color': DESIGN_TOKENS['glass']['border'],
}

GLASS_BUTTON_STYLE = {
    'fg_color': DESIGN_TOKENS['glass']['bg'],
    'hover_color': DESIGN_TOKENS['colors']['bg_hover'],
    'corner_radius': DESIGN_TOKENS['glass']['radius_sm'],
    'border_width': 1,
    'border_color': DESIGN_TOKENS['glass']['border'],
    'text_color': DESIGN_TOKENS['colors']['text_primary'],
}

GLASS_BUTTON_PRIMARY_STYLE = {
    'fg_color': DESIGN_TOKENS['colors']['accent_cyan_dim'],
    'hover_color': DESIGN_TOKENS['colors']['accent_cyan'],
    'corner_radius': DESIGN_TOKENS['glass']['radius_sm'],
    'border_width': 1,
    'border_color': DESIGN_TOKENS['colors']['accent_cyan'],
    'text_color': DESIGN_TOKENS['colors']['text_primary'],
}

GLASS_INPUT_STYLE = {
    'fg_color': DESIGN_TOKENS['glass']['bg_dark'],
    'corner_radius': DESIGN_TOKENS['glass']['radius_sm'],
    'border_width': 1,
    'border_color': DESIGN_TOKENS['glass']['border'],
    'text_color': DESIGN_TOKENS['colors']['text_primary'],
    'placeholder_text_color': DESIGN_TOKENS['colors']['text_muted'],
}

SIDEBAR_STYLE = {
    'fg_color': DESIGN_TOKENS['glass']['bg_dark'],
    'border_width': 1,
    'border_color': DESIGN_TOKENS['glass']['border'],
}

NAV_BUTTON_STYLE = {
    'fg_color': 'transparent',
    'hover_color': DESIGN_TOKENS['glass']['bg'],
    'corner_radius': DESIGN_TOKENS['glass']['radius_sm'],
    'text_color': DESIGN_TOKENS['colors']['text_secondary'],
    'text_color_hover': DESIGN_TOKENS['colors']['text_primary'],
}

NAV_BUTTON_ACTIVE_STYLE = {
    'fg_color': DESIGN_TOKENS['glass']['bg'],
    'corner_radius': DESIGN_TOKENS['glass']['radius_sm'],
    'border_width': 1,
    'border_color': DESIGN_TOKENS['glass']['border_active'],
    'text_color': DESIGN_TOKENS['colors']['accent_cyan'],
}


# =============================================================================
# ICONES MODERNES - Remplace les caracteres Unicode random
# =============================================================================

ICONS = {
    # Navigation
    'dashboard': '\u25C9',      # Cercle plein moderne
    'search': '\u2315',         # Loupe
    'chart': '\u2593',          # Graphique barre
    'portfolio': '\u25A3',      # Grille/Portfolio
    'alerts': '\u25C6',         # Diamant (alerte)
    'settings': '\u2699',       # Engrenage
    'user': '\u25CF',           # Cercle utilisateur
    'logout': '\u2192',         # Fleche sortie

    # Crypto
    'bitcoin': '\u20BF',        # Symbole Bitcoin
    'ethereum': '\u039E',       # Symbole Ethereum-like

    # Actions
    'add': '\u002B',            # Plus
    'remove': '\u2212',         # Moins
    'edit': '\u270E',           # Crayon
    'delete': '\u2715',         # Croix
    'check': '\u2713',          # Check
    'close': '\u2715',          # Fermer
    'menu': '\u2630',           # Hamburger menu
    'expand': '\u25BC',         # Fleche bas
    'collapse': '\u25B2',       # Fleche haut
    'refresh': '\u21BB',        # Rotation

    # Statuts
    'success': '\u2713',        # Check vert
    'warning': '\u26A0',        # Triangle warning
    'error': '\u2715',          # Croix rouge
    'info': '\u2139',           # Info

    # Trading
    'bullish': '\u25B2',        # Triangle haut
    'bearish': '\u25BC',        # Triangle bas
    'neutral': '\u25AC',        # Rectangle
}


# =============================================================================
# EXPORT SIMPLIFIE
# =============================================================================

# Alias pour import plus simple
COLORS = DESIGN_TOKENS['colors']
GLASS = DESIGN_TOKENS['glass']
SPACING = DESIGN_TOKENS['spacing']
ANIMATION = DESIGN_TOKENS['animation']
TYPOGRAPHY = DESIGN_TOKENS['typography']
