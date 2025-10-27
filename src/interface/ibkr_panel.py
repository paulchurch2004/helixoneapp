"""
Panneau IBKR Portfolio - Affichage en temps réel du portefeuille Interactive Brokers
"""

import customtkinter as ctk
import requests
import threading
import time
import math
from typing import Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger('helixone')

# URL de l'API Backend principal
API_BASE_URL = "http://127.0.0.1:8000"
# URL du microservice IBKR
IBKR_MICROSERVICE_URL = "http://127.0.0.1:8001"


class IBKRPortfolioPanel(ctk.CTkFrame):
    """Panneau d'affichage du portefeuille IBKR en temps réel"""

    def __init__(self, parent, auth_token: Optional[str] = None, **kwargs):
        super().__init__(parent, fg_color="#0d1117", **kwargs)

        self.auth_token = auth_token
        self.portfolio_data = None
        self.is_connected = False
        self.auto_refresh = True
        self.refresh_interval = 30  # secondes (augmenté pour éviter les timeouts)
        self.pulse_animation_running = False
        self.pulse_intensity = 0.5

        # Couleurs du thème
        self.COLORS = {
            'bg_primary': '#0d1117',
            'bg_secondary': '#161b22',
            'bg_tertiary': '#1c2128',
            'accent_green': '#2ea043',
            'accent_red': '#f85149',
            'accent_blue': '#58a6ff',
            'text_primary': '#f0f6fc',
            'text_secondary': '#8b949e',
            'border': '#30363d'
        }

        self._setup_ui()
        self._start_auto_refresh()

    def _setup_ui(self):
        """Créer l'interface utilisateur"""

        # En-tête du panneau
        header_frame = ctk.CTkFrame(self, fg_color=self.COLORS['bg_secondary'], corner_radius=10)
        header_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title_label = ctk.CTkLabel(
            header_frame,
            text="📊 Portefeuille IBKR",
            font=("Segoe UI", 24, "bold"),
            text_color=self.COLORS['text_primary']
        )
        title_label.pack(side="left", padx=20, pady=15)

        # Status de connexion
        self.status_label = ctk.CTkLabel(
            header_frame,
            text="● Déconnecté",
            font=("Segoe UI", 12),
            text_color=self.COLORS['accent_red']
        )
        self.status_label.pack(side="left", padx=10)

        # Bouton configurer
        config_btn = ctk.CTkButton(
            header_frame,
            text="⚙️ Configurer",
            command=self._show_config_dialog,
            fg_color=self.COLORS['bg_tertiary'],
            hover_color="#2a2d36",
            width=120,
            height=35,
            corner_radius=8
        )
        config_btn.pack(side="right", padx=10, pady=10)

        # Bouton rafraîchir
        refresh_btn = ctk.CTkButton(
            header_frame,
            text="🔄 Rafraîchir",
            command=self._refresh_portfolio,
            fg_color=self.COLORS['accent_blue'],
            hover_color="#4184e4",
            width=120,
            height=35,
            corner_radius=8
        )
        refresh_btn.pack(side="right", padx=10, pady=10)

        # Dernière mise à jour
        self.last_update_label = ctk.CTkLabel(
            header_frame,
            text="Dernière mise à jour: jamais",
            font=("Segoe UI", 10),
            text_color=self.COLORS['text_secondary']
        )
        self.last_update_label.pack(side="right", padx=10)

        # Frame principal avec scroll
        self.main_scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent"
        )
        self.main_scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Cartes de résumé (valeur, cash, P&L)
        self.summary_frame = ctk.CTkFrame(
            self.main_scroll_frame,
            fg_color="transparent"
        )
        self.summary_frame.pack(fill="x", pady=10)

        # Créer les cartes de métriques
        self.net_liq_card = self._create_metric_card(self.summary_frame, "Valeur Nette", "0.00 EUR", "💰")
        self.net_liq_card.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.cash_card = self._create_metric_card(self.summary_frame, "Liquidités", "0.00 EUR", "💵")
        self.cash_card.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        self.pnl_card = self._create_metric_card(self.summary_frame, "P&L Non Réalisé", "0.00 EUR", "📊")
        self.pnl_card.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

        # Configuration de la grille
        for i in range(3):
            self.summary_frame.grid_columnconfigure(i, weight=1)

        # Section Positions
        positions_header = ctk.CTkLabel(
            self.main_scroll_frame,
            text="📦 Positions",
            font=("Segoe UI", 18, "bold"),
            text_color=self.COLORS['text_primary'],
            anchor="w"
        )
        positions_header.pack(fill="x", pady=(20, 10))

        # Frame des positions
        self.positions_frame = ctk.CTkFrame(
            self.main_scroll_frame,
            fg_color=self.COLORS['bg_secondary'],
            corner_radius=10
        )
        self.positions_frame.pack(fill="both", expand=True, pady=10)

        # Section Alertes
        alerts_header = ctk.CTkLabel(
            self.main_scroll_frame,
            text="🔔 Alertes",
            font=("Segoe UI", 18, "bold"),
            text_color=self.COLORS['text_primary'],
            anchor="w"
        )
        alerts_header.pack(fill="x", pady=(20, 10))

        # Frame des alertes
        self.alerts_frame = ctk.CTkFrame(
            self.main_scroll_frame,
            fg_color=self.COLORS['bg_secondary'],
            corner_radius=10
        )
        self.alerts_frame.pack(fill="both", pady=10)

        # Message initial
        self._show_loading_message()

    def _create_metric_card(self, parent, title: str, value: str, icon: str) -> ctk.CTkFrame:
        """Créer une carte métrique"""
        card = ctk.CTkFrame(
            parent,
            fg_color=self.COLORS['bg_secondary'],
            corner_radius=10
        )

        # Icône
        icon_label = ctk.CTkLabel(
            card,
            text=icon,
            font=("Segoe UI", 32)
        )
        icon_label.pack(pady=(15, 5))

        # Titre
        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=("Segoe UI", 12),
            text_color=self.COLORS['text_secondary']
        )
        title_label.pack(pady=2)

        # Valeur
        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=("Segoe UI", 20, "bold"),
            text_color=self.COLORS['text_primary']
        )
        value_label.pack(pady=(2, 15))

        # Stocker la référence au label de valeur
        card.value_label = value_label

        return card

    def _show_loading_message(self):
        """Afficher un message de chargement"""
        # Démarrer l'animation de pulsation pendant le chargement
        self._start_pulse_animation()

        loading_label = ctk.CTkLabel(
            self.positions_frame,
            text="⏳ Connexion à IBKR...",
            font=("Segoe UI", 14),
            text_color=self.COLORS['text_secondary']
        )
        loading_label.pack(pady=50)

        # Charger les données
        self._refresh_portfolio()

    def _refresh_portfolio(self):
        """Rafraîchir les données du portefeuille"""
        def fetch_data():
            try:
                # Utiliser le microservice IBKR (pas d'auth nécessaire)
                endpoint = f"{IBKR_MICROSERVICE_URL}/dashboard"
                logger.info("Appel du microservice IBKR...")

                # Appeler le microservice IBKR
                response = requests.get(
                    endpoint,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    self.portfolio_data = data
                    self.is_connected = data.get('connected', False)
                    self._update_ui(data)
                else:
                    logger.error(f"Erreur microservice IBKR: {response.status_code}")
                    self._show_error(f"Erreur microservice: {response.status_code}")

            except requests.exceptions.ConnectionError:
                logger.error("Impossible de se connecter à l'API backend")
                self._show_error("API Backend non disponible")
            except Exception as e:
                logger.error(f"Erreur rafraîchissement IBKR: {e}")
                self._show_error(str(e))

        # Exécuter dans un thread pour ne pas bloquer l'UI
        thread = threading.Thread(target=fetch_data, daemon=True)
        thread.start()

    def _update_ui(self, data: Dict):
        """Mettre à jour l'interface avec les nouvelles données"""
        try:
            # Vérifier que le widget existe encore
            if not self.winfo_exists():
                logger.debug("Panneau IBKR détruit, arrêt de la mise à jour")
                self.auto_refresh = False
                return

            # Vérifier que data n'est pas None
            if not data:
                logger.error("Données vides reçues de l'API")
                self._show_error("Données invalides reçues")
                return

            # Si not connected, afficher le message d'erreur
            if not data.get('connected', False):
                error_msg = data.get('message', 'Connexion IBKR échouée')
                logger.warning(f"IBKR non connecté: {error_msg}")
                self._show_error(error_msg)
                # Démarrer l'animation de pulsation
                self._start_pulse_animation()
                return

            # Arrêter l'animation si elle tourne
            self._stop_pulse_animation()

            # Mettre à jour le status
            self.status_label.configure(
                text="● Connecté",
                text_color=self.COLORS['accent_green']
            )

            # Mettre à jour la dernière mise à jour
            now = datetime.now().strftime("%H:%M:%S")
            self.last_update_label.configure(text=f"Dernière mise à jour: {now}")

            # Mettre à jour les cartes de résumé
            portfolio = data.get('portfolio', {})

            net_liq = portfolio.get('net_liquidation', 0)
            currency = portfolio.get('currency', 'EUR')
            self.net_liq_card.value_label.configure(text=f"{net_liq:.2f} {currency}")

            cash = portfolio.get('total_cash', 0)
            self.cash_card.value_label.configure(text=f"{cash:.2f} {currency}")

            pnl = portfolio.get('unrealized_pnl', 0)
            pnl_color = self.COLORS['accent_green'] if pnl >= 0 else self.COLORS['accent_red']
            self.pnl_card.value_label.configure(
                text=f"{pnl:+.2f} {currency}",
                text_color=pnl_color
            )

            # Mettre à jour les positions
            self._update_positions(portfolio.get('positions', []))

            # Mettre à jour les alertes
            self._update_alerts(data.get('alerts', []))

        except Exception as e:
            logger.error(f"Erreur mise à jour UI IBKR: {e}")

    def _update_positions(self, positions: List[Dict]):
        """Mettre à jour la liste des positions"""
        # Nettoyer l'ancien contenu
        for widget in self.positions_frame.winfo_children():
            widget.destroy()

        if not positions:
            no_pos_label = ctk.CTkLabel(
                self.positions_frame,
                text="Aucune position",
                font=("Segoe UI", 14),
                text_color=self.COLORS['text_secondary']
            )
            no_pos_label.pack(pady=30)
            return

        # En-tête du tableau
        header_frame = ctk.CTkFrame(self.positions_frame, fg_color=self.COLORS['bg_tertiary'])
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        headers = ["Symbole", "Quantité", "Prix Moyen", "Prix Actuel", "P&L", "Valeur"]
        for i, header in enumerate(headers):
            label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=("Segoe UI", 11, "bold"),
                text_color=self.COLORS['text_secondary']
            )
            label.grid(row=0, column=i, padx=10, pady=8, sticky="w")

        # Lignes des positions
        for idx, pos in enumerate(positions):
            pos_frame = ctk.CTkFrame(
                self.positions_frame,
                fg_color=self.COLORS['bg_primary'] if idx % 2 == 0 else self.COLORS['bg_tertiary'],
                corner_radius=5
            )
            pos_frame.pack(fill="x", padx=10, pady=2)

            symbol = pos.get('symbol', 'N/A')
            quantity = pos.get('position', 0)
            avg_cost = pos.get('avg_cost', 0)
            market_price = pos.get('market_price', avg_cost)
            market_value = pos.get('market_value', 0)
            unrealized_pnl = pos.get('unrealized_pnl', 0)

            # Symbole
            ctk.CTkLabel(
                pos_frame,
                text=symbol,
                font=("Segoe UI", 12, "bold"),
                text_color=self.COLORS['text_primary']
            ).grid(row=0, column=0, padx=10, pady=8, sticky="w")

            # Quantité
            ctk.CTkLabel(
                pos_frame,
                text=f"{quantity:.0f}",
                font=("Segoe UI", 12),
                text_color=self.COLORS['text_primary']
            ).grid(row=0, column=1, padx=10, pady=8, sticky="w")

            # Prix moyen
            ctk.CTkLabel(
                pos_frame,
                text=f"{avg_cost:.2f}",
                font=("Segoe UI", 12),
                text_color=self.COLORS['text_secondary']
            ).grid(row=0, column=2, padx=10, pady=8, sticky="w")

            # Prix actuel
            ctk.CTkLabel(
                pos_frame,
                text=f"{market_price:.2f}",
                font=("Segoe UI", 12),
                text_color=self.COLORS['text_primary']
            ).grid(row=0, column=3, padx=10, pady=8, sticky="w")

            # P&L
            pnl_color = self.COLORS['accent_green'] if unrealized_pnl >= 0 else self.COLORS['accent_red']
            ctk.CTkLabel(
                pos_frame,
                text=f"{unrealized_pnl:+.2f}",
                font=("Segoe UI", 12, "bold"),
                text_color=pnl_color
            ).grid(row=0, column=4, padx=10, pady=8, sticky="w")

            # Valeur
            ctk.CTkLabel(
                pos_frame,
                text=f"{market_value:.2f}",
                font=("Segoe UI", 12),
                text_color=self.COLORS['text_primary']
            ).grid(row=0, column=5, padx=10, pady=8, sticky="w")

    def _update_alerts(self, alerts: List[Dict]):
        """Mettre à jour les alertes"""
        # Nettoyer l'ancien contenu
        for widget in self.alerts_frame.winfo_children():
            widget.destroy()

        if not alerts:
            no_alert_label = ctk.CTkLabel(
                self.alerts_frame,
                text="✅ Aucune alerte",
                font=("Segoe UI", 14),
                text_color=self.COLORS['accent_green']
            )
            no_alert_label.pack(pady=30)
            return

        # Afficher chaque alerte
        for alert in alerts:
            alert_frame = ctk.CTkFrame(
                self.alerts_frame,
                fg_color=self.COLORS['bg_tertiary'],
                corner_radius=8
            )
            alert_frame.pack(fill="x", padx=10, pady=5)

            severity = alert.get('severity', 'low')
            title = alert.get('title', 'Alerte')
            message = alert.get('message', '')

            # Couleur selon la sévérité
            severity_colors = {
                'low': self.COLORS['accent_blue'],
                'medium': '#f0883e',
                'high': '#f85149',
                'critical': '#d62839'
            }
            color = severity_colors.get(severity, self.COLORS['accent_blue'])

            # Icône selon la sévérité
            severity_icons = {
                'low': 'ℹ️',
                'medium': '⚠️',
                'high': '🚨',
                'critical': '🔴'
            }
            icon = severity_icons.get(severity, 'ℹ️')

            # Titre de l'alerte
            title_label = ctk.CTkLabel(
                alert_frame,
                text=f"{icon} {title}",
                font=("Segoe UI", 13, "bold"),
                text_color=color
            )
            title_label.pack(anchor="w", padx=15, pady=(10, 5))

            # Message
            if message:
                msg_label = ctk.CTkLabel(
                    alert_frame,
                    text=message,
                    font=("Segoe UI", 11),
                    text_color=self.COLORS['text_secondary'],
                    wraplength=600
                )
                msg_label.pack(anchor="w", padx=15, pady=(0, 10))

    def _show_auth_required(self):
        """Afficher un message demandant l'authentification"""
        try:
            # Vérifier que le widget existe
            if not self.winfo_exists():
                return

            # Démarrer l'animation de pulsation
            self._start_pulse_animation()

            self.status_label.configure(
                text="● Non connecté",
                text_color=self.COLORS['accent_red']
            )

            # Nettoyer les positions
            for widget in self.positions_frame.winfo_children():
                widget.destroy()

            # Créer un frame d'information
            info_frame = ctk.CTkFrame(
                self.positions_frame,
                fg_color=self.COLORS['bg_tertiary'],
                corner_radius=10
            )
            info_frame.pack(fill="both", expand=True, padx=20, pady=20)

            # Icône
            icon_label = ctk.CTkLabel(
                info_frame,
                text="🔐",
                font=("Segoe UI", 64)
            )
            icon_label.pack(pady=(30, 10))

            # Titre
            title_label = ctk.CTkLabel(
                info_frame,
                text="Authentification requise",
                font=("Segoe UI", 20, "bold"),
                text_color=self.COLORS['text_primary']
            )
            title_label.pack(pady=(0, 10))

            # Message
            message_label = ctk.CTkLabel(
                info_frame,
                text="Pour afficher votre portefeuille IBKR en temps réel,\nveuillez vous connecter à votre compte HelixOne.",
                font=("Segoe UI", 13),
                text_color=self.COLORS['text_secondary'],
                justify="center"
            )
            message_label.pack(pady=(0, 30))

            # Nettoyer les alertes aussi
            for widget in self.alerts_frame.winfo_children():
                widget.destroy()

            info_label = ctk.CTkLabel(
                self.alerts_frame,
                text="ℹ️ Connexion requise pour voir les alertes",
                font=("Segoe UI", 12),
                text_color=self.COLORS['text_secondary']
            )
            info_label.pack(pady=20)

        except Exception as e:
            logger.error(f"Erreur affichage auth required IBKR: {e}")

    def _show_error(self, error_msg: str):
        """Afficher un message d'erreur"""
        try:
            # Vérifier que le widget existe
            if not self.winfo_exists():
                return

            # Démarrer l'animation de pulsation
            self._start_pulse_animation()

            self.status_label.configure(
                text="● Erreur",
                text_color=self.COLORS['accent_red']
            )

            # Nettoyer les positions
            for widget in self.positions_frame.winfo_children():
                widget.destroy()

            error_label = ctk.CTkLabel(
                self.positions_frame,
                text=f"❌ {error_msg}",
                font=("Segoe UI", 14),
                text_color=self.COLORS['accent_red']
            )
            error_label.pack(pady=50)
        except Exception as e:
            logger.error(f"Erreur affichage erreur IBKR: {e}")

    def _start_auto_refresh(self):
        """Démarrer le rafraîchissement automatique"""
        def auto_refresh_loop():
            while self.auto_refresh:
                time.sleep(self.refresh_interval)
                if self.auto_refresh and self.winfo_exists():
                    self._refresh_portfolio()
                elif not self.winfo_exists():
                    # Widget détruit, arrêter le thread
                    self.auto_refresh = False
                    break

        thread = threading.Thread(target=auto_refresh_loop, daemon=True)
        thread.start()

    def stop_auto_refresh(self):
        """Arrêter le rafraîchissement automatique"""
        self.auto_refresh = False

    def _pulse_disconnected_status(self):
        """Animation de pulsation pour le status déconnecté"""
        if not self.pulse_animation_running:
            return

        try:
            if not self.winfo_exists():
                self.pulse_animation_running = False
                return

            # Calculer la nouvelle intensité (oscillation entre 0.3 et 1.0)
            self.pulse_intensity += 0.05
            intensity = 0.5 + 0.5 * abs(math.sin(self.pulse_intensity))

            # Convertir l'intensité en couleur RGB
            # Rouge de base: #f85149
            base_r, base_g, base_b = 248, 81, 73

            # Appliquer l'intensité
            r = int(base_r * intensity)
            g = int(base_g * intensity)
            b = int(base_b * intensity)

            # Créer la couleur au format hex
            color = f"#{r:02x}{g:02x}{b:02x}"

            # Mettre à jour la couleur du label
            self.status_label.configure(text_color=color)

            # Continuer l'animation
            if self.pulse_animation_running:
                self.after(50, self._pulse_disconnected_status)

        except Exception as e:
            logger.error(f"Erreur animation pulse: {e}")
            self.pulse_animation_running = False

    def _start_pulse_animation(self):
        """Démarrer l'animation de pulsation"""
        if not self.pulse_animation_running:
            self.pulse_animation_running = True
            self._pulse_disconnected_status()

    def _stop_pulse_animation(self):
        """Arrêter l'animation de pulsation"""
        self.pulse_animation_running = False

    def _show_config_dialog(self):
        """Afficher la boîte de dialogue de configuration IBKR"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Configuration IBKR")
        dialog.geometry("600x550")
        dialog.resizable(False, False)

        # Centrer la fenêtre
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        # Configuration du thème
        dialog.configure(fg_color=self.COLORS['bg_primary'])

        # Frame principal avec padding
        main_frame = ctk.CTkFrame(dialog, fg_color=self.COLORS['bg_secondary'], corner_radius=10)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Titre
        title_label = ctk.CTkLabel(
            main_frame,
            text="⚙️ Configuration Interactive Brokers",
            font=("Segoe UI", 20, "bold"),
            text_color=self.COLORS['text_primary']
        )
        title_label.pack(pady=(20, 10))

        # Description
        desc_label = ctk.CTkLabel(
            main_frame,
            text="Configurez votre connexion à Interactive Brokers TWS/Gateway",
            font=("Segoe UI", 12),
            text_color=self.COLORS['text_secondary']
        )
        desc_label.pack(pady=(0, 20))

        # Frame pour les champs de saisie
        fields_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        fields_frame.pack(fill="x", padx=30, pady=10)

        # Host
        host_label = ctk.CTkLabel(
            fields_frame,
            text="Host:",
            font=("Segoe UI", 12),
            text_color=self.COLORS['text_primary']
        )
        host_label.pack(anchor="w", pady=(10, 5))

        host_entry = ctk.CTkEntry(
            fields_frame,
            placeholder_text="127.0.0.1",
            height=40,
            font=("Segoe UI", 12),
            fg_color=self.COLORS['bg_tertiary'],
            border_color=self.COLORS['border']
        )
        host_entry.pack(fill="x", pady=(0, 10))
        host_entry.insert(0, "127.0.0.1")

        # Port
        port_label = ctk.CTkLabel(
            fields_frame,
            text="Port:",
            font=("Segoe UI", 12),
            text_color=self.COLORS['text_primary']
        )
        port_label.pack(anchor="w", pady=(10, 5))

        port_entry = ctk.CTkEntry(
            fields_frame,
            placeholder_text="7496 (LIVE) ou 7497 (PAPER)",
            height=40,
            font=("Segoe UI", 12),
            fg_color=self.COLORS['bg_tertiary'],
            border_color=self.COLORS['border']
        )
        port_entry.pack(fill="x", pady=(0, 5))
        port_entry.insert(0, "7497")

        # Info port
        port_info = ctk.CTkLabel(
            fields_frame,
            text="💡 7496 = LIVE Trading | 7497 = Paper Trading",
            font=("Segoe UI", 10),
            text_color=self.COLORS['accent_blue']
        )
        port_info.pack(anchor="w", pady=(0, 10))

        # Client ID
        client_label = ctk.CTkLabel(
            fields_frame,
            text="Client ID:",
            font=("Segoe UI", 12),
            text_color=self.COLORS['text_primary']
        )
        client_label.pack(anchor="w", pady=(10, 5))

        client_entry = ctk.CTkEntry(
            fields_frame,
            placeholder_text="2",
            height=40,
            font=("Segoe UI", 12),
            fg_color=self.COLORS['bg_tertiary'],
            border_color=self.COLORS['border']
        )
        client_entry.pack(fill="x", pady=(0, 5))
        client_entry.insert(0, "2")

        # Info client ID
        client_info = ctk.CTkLabel(
            fields_frame,
            text="💡 Identifiant unique de connexion (1-999)",
            font=("Segoe UI", 10),
            text_color=self.COLORS['text_secondary']
        )
        client_info.pack(anchor="w", pady=(0, 10))

        # Status de connexion
        status_frame = ctk.CTkFrame(main_frame, fg_color=self.COLORS['bg_tertiary'], corner_radius=8)
        status_frame.pack(fill="x", padx=30, pady=20)

        status_label = ctk.CTkLabel(
            status_frame,
            text="",
            font=("Segoe UI", 11),
            text_color=self.COLORS['text_secondary']
        )
        status_label.pack(pady=10)

        # Frame pour les boutons
        buttons_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=30, pady=(10, 20))

        def test_connection():
            """Tester la connexion IBKR"""
            status_label.configure(text="⏳ Test de connexion en cours...", text_color=self.COLORS['accent_blue'])
            dialog.update()

            def test():
                try:
                    host = host_entry.get() or "127.0.0.1"
                    port = int(port_entry.get() or "7497")
                    client_id = int(client_entry.get() or "2")

                    # Appeler le microservice pour tester la connexion
                    response = requests.post(
                        f"{IBKR_MICROSERVICE_URL}/connect",
                        json={
                            "host": host,
                            "port": port,
                            "client_id": client_id
                        },
                        timeout=30
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get('connected'):
                            status_label.configure(
                                text="✅ Connexion réussie!",
                                text_color=self.COLORS['accent_green']
                            )
                            # Rafraîchir le panneau principal
                            self._refresh_portfolio()
                        else:
                            error_msg = data.get('message', 'Erreur de connexion')
                            status_label.configure(
                                text=f"❌ {error_msg}",
                                text_color=self.COLORS['accent_red']
                            )
                    else:
                        status_label.configure(
                            text=f"❌ Erreur serveur: {response.status_code}",
                            text_color=self.COLORS['accent_red']
                        )

                except ValueError:
                    status_label.configure(
                        text="❌ Port et Client ID doivent être des nombres",
                        text_color=self.COLORS['accent_red']
                    )
                except requests.exceptions.ConnectionError:
                    status_label.configure(
                        text="❌ Microservice IBKR non disponible",
                        text_color=self.COLORS['accent_red']
                    )
                except Exception as e:
                    logger.error(f"Erreur test connexion: {e}")
                    status_label.configure(
                        text=f"❌ Erreur: {str(e)}",
                        text_color=self.COLORS['accent_red']
                    )

            # Exécuter dans un thread
            thread = threading.Thread(target=test, daemon=True)
            thread.start()

        def save_and_connect():
            """Sauvegarder et connecter"""
            # Pour l'instant, on teste juste la connexion
            # TODO: Implémenter la sauvegarde de la configuration
            test_connection()

        # Bouton Tester
        test_btn = ctk.CTkButton(
            buttons_frame,
            text="🔌 Tester la connexion",
            command=test_connection,
            fg_color=self.COLORS['accent_blue'],
            hover_color="#4184e4",
            height=40,
            font=("Segoe UI", 12, "bold"),
            corner_radius=8
        )
        test_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # Bouton Sauvegarder
        save_btn = ctk.CTkButton(
            buttons_frame,
            text="💾 Sauvegarder et Connecter",
            command=save_and_connect,
            fg_color=self.COLORS['accent_green'],
            hover_color="#26843b",
            height=40,
            font=("Segoe UI", 12, "bold"),
            corner_radius=8
        )
        save_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Bouton Annuler
        cancel_btn = ctk.CTkButton(
            main_frame,
            text="Annuler",
            command=dialog.destroy,
            fg_color=self.COLORS['bg_tertiary'],
            hover_color="#2a2d36",
            height=35,
            font=("Segoe UI", 11),
            corner_radius=8
        )
        cancel_btn.pack(pady=(0, 20))

    def destroy(self):
        """Nettoyer avant destruction du widget"""
        logger.debug("Destruction du panneau IBKR")
        self.stop_auto_refresh()
        self._stop_pulse_animation()
        super().destroy()
