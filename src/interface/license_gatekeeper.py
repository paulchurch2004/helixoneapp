"""
License Gatekeeper - Contrôle d'accès aux features PRO

Gère la vérification des licences et l'affichage des modals d'upgrade.

Tiers:
- trial: Fonctionnalités de base limitées
- basic: Fonctionnalités de base complètes
- premium: Fonctionnalités PRO (99€/mois)
- professional: Toutes fonctionnalités (499€/mois)

Features:
- basic_optimization: Optimisation Markowitz basique
- advanced_risk: VaR, CVaR, Sharpe, Sortino, etc.
- black_litterman: Optimisation Black-Litterman
- rl_optimization: Optimisation par Reinforcement Learning
- attribution: Attribution de performance Brinson
"""

import logging
import customtkinter as ctk
from typing import Optional, List, Literal
from datetime import datetime

logger = logging.getLogger(__name__)


# Définition des features par tier
FEATURE_TIERS = {
    'basic_optimization': ['trial', 'basic', 'premium', 'professional'],
    'ml_predictions': ['trial', 'basic', 'premium', 'professional'],
    'basic_charts': ['trial', 'basic', 'premium', 'professional'],
    'watchlist': ['trial', 'basic', 'premium', 'professional'],

    # PRO Features
    'advanced_risk': ['premium', 'professional'],
    'black_litterman': ['premium', 'professional'],
    'attribution': ['premium', 'professional'],
    'stress_testing': ['premium', 'professional'],
    'custom_scenarios': ['premium', 'professional'],

    # INSTITUTIONAL Features
    'rl_optimization': ['professional'],
    'optimal_execution': ['professional'],
    'market_making': ['professional'],
    'api_access': ['professional'],
    'multi_user': ['professional'],
}


# Pricing information
PRICING = {
    'trial': {
        'price': '0€',
        'period': '7 jours',
        'features_count': 'Limité',
        'analyses_per_day': 5,
    },
    'basic': {
        'price': '49€',
        'period': '/mois',
        'features_count': 'Basiques',
        'analyses_per_day': 50,
    },
    'premium': {
        'price': '99€',
        'period': '/mois',
        'features_count': 'PRO',
        'analyses_per_day': 200,
        'highlight': True,
    },
    'professional': {
        'price': '499€',
        'period': '/mois',
        'features_count': 'Toutes',
        'analyses_per_day': 'Illimité',
    },
}


class FeatureGatekeeper:
    """Contrôle d'accès aux features basé sur la licence"""

    _cached_license = None
    _cache_time = None
    _cache_ttl = 300  # 5 minutes

    @classmethod
    def get_license_status(cls, api_client) -> dict:
        """
        Récupère le statut de licence (avec cache)

        Returns:
            {
                'license_type': 'premium',
                'status': 'active',
                'expires_at': '2024-12-31',
                'features': {...},
                'quota_daily_analyses': 200
            }
        """
        now = datetime.now()

        # Utiliser le cache si valide
        if (cls._cached_license and cls._cache_time and
                (now - cls._cache_time).total_seconds() < cls._cache_ttl):
            return cls._cached_license

        try:
            license_info = api_client.get_license_status()
            cls._cached_license = license_info
            cls._cache_time = now
            return license_info
        except Exception as e:
            logger.error(f"Erreur récupération licence: {e}")
            # Retourner une licence trial par défaut en cas d'erreur
            return {
                'license_type': 'trial',
                'status': 'active',
                'expires_at': None,
                'features': {},
                'quota_daily_analyses': 5
            }

    @classmethod
    def invalidate_cache(cls):
        """Invalide le cache de licence (après upgrade par exemple)"""
        cls._cached_license = None
        cls._cache_time = None

    @staticmethod
    def check_feature_access(feature: str, api_client) -> bool:
        """
        Vérifie si l'utilisateur a accès à une feature

        Args:
            feature: Nom de la feature (ex: 'advanced_risk')
            api_client: Client API HelixOne

        Returns:
            True si accès autorisé, False sinon
        """
        license = FeatureGatekeeper.get_license_status(api_client)
        license_type = license.get('license_type', 'trial')

        # Vérifier dans la map des features
        allowed_tiers = FEATURE_TIERS.get(feature, [])

        has_access = license_type in allowed_tiers

        logger.info(
            f"Feature '{feature}' - Licence: {license_type} - "
            f"Accès: {'✓' if has_access else '✗'}"
        )

        return has_access

    @staticmethod
    def get_required_tier(feature: str) -> List[str]:
        """Retourne les tiers requis pour une feature"""
        return FEATURE_TIERS.get(feature, ['professional'])

    @staticmethod
    def show_upgrade_modal(parent, feature_name: str, api_client):
        """
        Affiche un modal d'upgrade pour une feature locked

        Args:
            parent: Widget parent
            feature_name: Nom de la feature
            api_client: Client API
        """
        license = FeatureGatekeeper.get_license_status(api_client)
        current_tier = license.get('license_type', 'trial')
        required_tiers = FeatureGatekeeper.get_required_tier(feature_name)

        # Déterminer le tier minimum requis
        if 'premium' in required_tiers:
            min_tier = 'premium'
        elif 'professional' in required_tiers:
            min_tier = 'professional'
        else:
            min_tier = 'professional'

        # Créer le modal
        modal = UpgradeModal(
            parent,
            feature_name=feature_name,
            current_tier=current_tier,
            required_tier=min_tier
        )

    @staticmethod
    def create_locked_banner(parent, feature_name: str, api_client) -> ctk.CTkFrame:
        """
        Crée un banner "Feature Locked" avec bouton upgrade

        Args:
            parent: Widget parent
            feature_name: Nom de la feature
            api_client: Client API

        Returns:
            Frame avec banner

        Usage:
            if not FeatureGatekeeper.check_feature_access('advanced_risk', client):
                banner = FeatureGatekeeper.create_locked_banner(self, 'advanced_risk', client)
                banner.pack(fill="x", padx=20, pady=10)
        """
        from .design_system import DESIGN_TOKENS

        banner = ctk.CTkFrame(
            parent,
            fg_color=DESIGN_TOKENS['GLASS']['bg_dark'],
            corner_radius=12,
            border_width=1,
            border_color=DESIGN_TOKENS['COLORS']['accent_cyan']
        )

        # Icône lock
        lock_label = ctk.CTkLabel(
            banner,
            text="🔒",
            font=("Arial", 32)
        )
        lock_label.pack(side="left", padx=20, pady=15)

        # Texte info
        info_frame = ctk.CTkFrame(banner, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True, padx=10, pady=15)

        title = ctk.CTkLabel(
            info_frame,
            text=f"Feature PRO : {feature_name.replace('_', ' ').title()}",
            font=("Arial", 14, "bold"),
            text_color=DESIGN_TOKENS['COLORS']['accent_cyan']
        )
        title.pack(anchor="w")

        description = ctk.CTkLabel(
            info_frame,
            text="Cette fonctionnalité nécessite une licence PREMIUM ou PROFESSIONAL",
            font=("Arial", 11),
            text_color=DESIGN_TOKENS['COLORS']['text_secondary']
        )
        description.pack(anchor="w", pady=(5, 0))

        # Bouton upgrade
        def on_upgrade():
            FeatureGatekeeper.show_upgrade_modal(parent, feature_name, api_client)

        upgrade_btn = ctk.CTkButton(
            banner,
            text="Upgrade Now →",
            font=("Arial", 12, "bold"),
            fg_color=DESIGN_TOKENS['COLORS']['accent_cyan'],
            hover_color=DESIGN_TOKENS['COLORS']['accent_blue'],
            corner_radius=8,
            width=150,
            height=40,
            command=on_upgrade
        )
        upgrade_btn.pack(side="right", padx=20, pady=15)

        return banner


class UpgradeModal(ctk.CTkToplevel):
    """Modal d'upgrade avec pricing"""

    def __init__(self, parent, feature_name: str, current_tier: str, required_tier: str):
        super().__init__(parent)

        self.title("Upgrade to PRO")
        self.geometry("700x500")
        self.resizable(False, False)

        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (700 // 2)
        y = (self.winfo_screenheight() // 2) - (500 // 2)
        self.geometry(f"700x500+{x}+{y}")

        from .design_system import DESIGN_TOKENS

        # Background
        self.configure(fg_color=DESIGN_TOKENS['GLASS']['bg_dark'])

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent", height=80)
        header.pack(fill="x", padx=30, pady=(20, 10))

        title_label = ctk.CTkLabel(
            header,
            text="🚀 Unlock PRO Features",
            font=("Arial", 24, "bold"),
            text_color=DESIGN_TOKENS['COLORS']['accent_cyan']
        )
        title_label.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            header,
            text=f"'{feature_name.replace('_', ' ').title()}' requires {required_tier.upper()} tier",
            font=("Arial", 12),
            text_color=DESIGN_TOKENS['COLORS']['text_secondary']
        )
        subtitle.pack(anchor="w", pady=(5, 0))

        # Pricing cards
        pricing_frame = ctk.CTkFrame(self, fg_color="transparent")
        pricing_frame.pack(fill="both", expand=True, padx=30, pady=10)

        # PREMIUM Card
        self._create_pricing_card(
            pricing_frame,
            tier='premium',
            is_current=(current_tier == 'premium'),
            is_required=(required_tier == 'premium'),
            side="left"
        )

        # Spacer
        ctk.CTkFrame(pricing_frame, width=20, fg_color="transparent").pack(side="left")

        # PROFESSIONAL Card
        self._create_pricing_card(
            pricing_frame,
            tier='professional',
            is_current=(current_tier == 'professional'),
            is_required=(required_tier == 'professional'),
            side="left"
        )

        # Footer
        footer = ctk.CTkFrame(self, fg_color="transparent", height=60)
        footer.pack(fill="x", padx=30, pady=20)

        close_btn = ctk.CTkButton(
            footer,
            text="Close",
            font=("Arial", 12),
            fg_color=DESIGN_TOKENS['GLASS']['bg_light'],
            hover_color=DESIGN_TOKENS['GLASS']['border'],
            width=120,
            command=self.destroy
        )
        close_btn.pack(side="right")

    def _create_pricing_card(self, parent, tier: str, is_current: bool, is_required: bool, side: str):
        """Crée une carte de pricing"""
        from .design_system import DESIGN_TOKENS

        info = PRICING[tier]
        highlight = info.get('highlight', False) or is_required

        # Card frame
        card = ctk.CTkFrame(
            parent,
            fg_color=DESIGN_TOKENS['GLASS']['bg_light'] if highlight else DESIGN_TOKENS['GLASS']['bg_dark'],
            corner_radius=12,
            border_width=2 if highlight else 1,
            border_color=DESIGN_TOKENS['COLORS']['accent_cyan'] if highlight else DESIGN_TOKENS['GLASS']['border']
        )
        card.pack(side=side, fill="both", expand=True)

        # Header
        header = ctk.CTkFrame(card, fg_color="transparent", height=60)
        header.pack(fill="x", padx=20, pady=(20, 10))

        tier_label = ctk.CTkLabel(
            header,
            text=tier.upper(),
            font=("Arial", 16, "bold"),
            text_color=DESIGN_TOKENS['COLORS']['accent_cyan'] if highlight else DESIGN_TOKENS['COLORS']['text_primary']
        )
        tier_label.pack(anchor="w")

        if is_current:
            current_badge = ctk.CTkLabel(
                header,
                text="CURRENT",
                font=("Arial", 10, "bold"),
                fg_color=DESIGN_TOKENS['COLORS']['accent_green'],
                corner_radius=4,
                padx=8,
                pady=2
            )
            current_badge.pack(anchor="w", pady=(5, 0))

        if is_required:
            required_badge = ctk.CTkLabel(
                header,
                text="REQUIRED",
                font=("Arial", 10, "bold"),
                fg_color=DESIGN_TOKENS['COLORS']['accent_blue'],
                corner_radius=4,
                padx=8,
                pady=2
            )
            required_badge.pack(anchor="w", pady=(5, 0))

        # Price
        price_label = ctk.CTkLabel(
            card,
            text=info['price'],
            font=("Arial", 32, "bold"),
            text_color=DESIGN_TOKENS['COLORS']['text_primary']
        )
        price_label.pack(pady=(10, 0))

        period_label = ctk.CTkLabel(
            card,
            text=info['period'],
            font=("Arial", 12),
            text_color=DESIGN_TOKENS['COLORS']['text_secondary']
        )
        period_label.pack()

        # Features
        features_frame = ctk.CTkFrame(card, fg_color="transparent")
        features_frame.pack(fill="both", expand=True, padx=20, pady=15)

        features = self._get_tier_features(tier)
        for feature in features:
            feature_row = ctk.CTkFrame(features_frame, fg_color="transparent")
            feature_row.pack(fill="x", pady=3)

            ctk.CTkLabel(
                feature_row,
                text="✓",
                font=("Arial", 12),
                text_color=DESIGN_TOKENS['COLORS']['accent_cyan'],
                width=20
            ).pack(side="left")

            ctk.CTkLabel(
                feature_row,
                text=feature,
                font=("Arial", 11),
                text_color=DESIGN_TOKENS['COLORS']['text_secondary'],
                anchor="w"
            ).pack(side="left", fill="x", expand=True)

        # CTA Button
        if not is_current:
            cta_btn = ctk.CTkButton(
                card,
                text=f"Upgrade to {tier.upper()}",
                font=("Arial", 12, "bold"),
                fg_color=DESIGN_TOKENS['COLORS']['accent_cyan'] if highlight else DESIGN_TOKENS['COLORS']['accent_blue'],
                hover_color=DESIGN_TOKENS['COLORS']['accent_blue'],
                height=40,
                corner_radius=8,
                command=lambda: self._on_upgrade(tier)
            )
            cta_btn.pack(fill="x", padx=20, pady=20)

    def _get_tier_features(self, tier: str) -> List[str]:
        """Retourne la liste des features pour un tier"""
        features_map = {
            'premium': [
                "Advanced Risk Metrics",
                "Black-Litterman Optimization",
                "Performance Attribution",
                "Stress Testing",
                "200 analyses/jour",
                "Support prioritaire"
            ],
            'professional': [
                "All PREMIUM features",
                "RL-Based Optimization",
                "Optimal Execution",
                "Market Making Strategies",
                "API Access",
                "Analyses illimitées",
                "Multi-utilisateurs",
                "Support dédié 24/7"
            ]
        }
        return features_map.get(tier, [])

    def _on_upgrade(self, tier: str):
        """Callback upgrade button"""
        logger.info(f"Upgrade request: {tier}")
        # TODO: Intégrer avec système de paiement Stripe
        # Pour l'instant, juste fermer le modal
        self.destroy()


# ============================================================================
# HELPERS
# ============================================================================

def requires_feature(feature: str):
    """
    Décorateur pour méthodes qui nécessitent une feature

    Usage:
        @requires_feature('advanced_risk')
        def show_advanced_risk_panel(self):
            # Cette méthode ne sera appelée que si feature accessible
            pass
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'api_client'):
                if FeatureGatekeeper.check_feature_access(feature, self.api_client):
                    return func(self, *args, **kwargs)
                else:
                    FeatureGatekeeper.show_upgrade_modal(self, feature, self.api_client)
                    return None
            else:
                logger.warning(f"Méthode {func.__name__} nécessite self.api_client")
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Tests
    print("🧪 Tests License Gatekeeper\n")

    # Mock API client
    class MockClient:
        def get_license_status(self):
            return {
                'license_type': 'trial',
                'status': 'active',
                'expires_at': '2024-12-31',
                'features': {},
                'quota_daily_analyses': 5
            }

    client = MockClient()

    # Test accès features
    print(f"basic_optimization (trial): {FeatureGatekeeper.check_feature_access('basic_optimization', client)}")
    print(f"advanced_risk (trial): {FeatureGatekeeper.check_feature_access('advanced_risk', client)}")
    print(f"rl_optimization (trial): {FeatureGatekeeper.check_feature_access('rl_optimization', client)}")

    print("\n✅ Tests passés!")
