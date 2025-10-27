"""
Interface d'activation et de gestion des licences HelixOne
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
from datetime import datetime
from typing import Optional, Dict, Any

from src.security.license_manager import get_license_manager
from src.utils.logger import get_logger


# =========================
#   Boîte de dialogue Licence
# =========================
class LicenseDialog(ctk.CTkToplevel):
    """Dialogue d'activation de licence"""

    def __init__(self, parent, force_activation: bool = False):
        super().__init__(parent)

        self.license_manager = get_license_manager()
        self.logger = get_logger()
        self.force_activation = force_activation
        self.result: Optional[Dict[str, Any]] = None

        # Configuration fenêtre
        self.title("HelixOne - Gestion des Licences")
        self.geometry("500x600")
        self.resizable(False, False)
        self.configure(fg_color="#1c1f26")

        # Modal
        self.transient(parent)
        self.grab_set()

        # Centrer la fenêtre
        self.center_window()

        # Interface
        self.setup_ui()

        # Vérifier licence actuelle
        self.update_license_status_async()

        # Si activation forcée, on limite certaines actions
        if self.force_activation:
            self.protocol("WM_DELETE_WINDOW", self.on_forced_close_block)
        else:
            self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- helpers UI ----------
    def center_window(self):
        """Centre la fenêtre sur l'écran"""
        self.update_idletasks()
        width = self.winfo_width() or 500
        height = self.winfo_height() or 600
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="#2a2d36")
        header_frame.pack(fill="x", padx=20, pady=(20, 0))

        title = ctk.CTkLabel(
            header_frame,
            text="🔐 Gestion des Licences HelixOne",
            font=("Arial", 20, "bold"),
            text_color="#00BFFF"
        )
        title.pack(pady=15)

        # Status actuel
        self.status_frame = ctk.CTkFrame(self, fg_color="#161920")
        self.status_frame.pack(fill="x", padx=20, pady=10)

        status_label = ctk.CTkLabel(
            self.status_frame,
            text="📊 Statut Actuel",
            font=("Arial", 14, "bold"),
            text_color="#00BFFF"
        )
        status_label.pack(pady=(10, 5))

        self.status_text = ctk.CTkLabel(
            self.status_frame,
            text="Vérification en cours...",
            font=("Arial", 12),
            text_color="#CCCCCC",
            justify="left"
        )
        self.status_text.pack(pady=(0, 10), padx=10, anchor="w")

        # Informations licence
        self.info_frame = ctk.CTkFrame(self, fg_color="#161920")
        self.info_frame.pack(fill="x", padx=20, pady=10)

        self.info_rows: Dict[str, ctk.CTkLabel] = {}
        for label in ("Type", "Email", "Clé", "Statut", "Activée le", "Expire le", "Jours restants"):
            row = ctk.CTkFrame(self.info_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=3)
            ctk.CTkLabel(row, text=f"{label} :", font=("Arial", 12, "bold"), text_color="#AAAAAA", width=120, anchor="w").pack(side="left")
            val = ctk.CTkLabel(row, text="-", font=("Arial", 12), text_color="#DDDDDD")
            val.pack(side="left")
            self.info_rows[label] = val

        # Zone d'activation
        activation_frame = ctk.CTkFrame(self, fg_color="#2a2d36")
        activation_frame.pack(fill="x", padx=20, pady=10)

        activation_label = ctk.CTkLabel(
            activation_frame,
            text="🔑 Activation de Licence",
            font=("Arial", 14, "bold"),
            text_color="#00BFFF"
        )
        activation_label.pack(pady=(15, 10))

        # Email
        email_frame = ctk.CTkFrame(activation_frame, fg_color="transparent")
        email_frame.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            email_frame,
            text="Email :",
            font=("Arial", 12),
            text_color="#AAAAAA"
        ).pack(anchor="w")

        self.email_entry = ctk.CTkEntry(
            email_frame,
            placeholder_text="votre.email@example.com",
            font=("Arial", 12),
            height=35
        )
        self.email_entry.pack(fill="x", pady=(5, 0))

        # Clé de licence
        key_frame = ctk.CTkFrame(activation_frame, fg_color="transparent")
        key_frame.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            key_frame,
            text="Clé de licence :",
            font=("Arial", 12),
            text_color="#AAAAAA"
        ).pack(anchor="w")

        self.license_key_entry = ctk.CTkEntry(
            key_frame,
            placeholder_text="PREMIUM-XXXX-XXXX-XXXX-XXXX",
            font=("Arial", 12),
            height=35
        )
        self.license_key_entry.pack(fill="x", pady=(5, 0))

        # Boutons d'activation
        buttons_frame = ctk.CTkFrame(activation_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=20, pady=15)

        self.activate_button = ctk.CTkButton(
            buttons_frame,
            text="🔓 Activer la Licence",
            font=("Arial", 12, "bold"),
            fg_color="#00FF88",
            hover_color="#00CC6A",
            command=self.activate_license
        )
        self.activate_button.pack(side="left", padx=(0, 10))

        self.trial_button = ctk.CTkButton(
            buttons_frame,
            text="🆓 Essai Gratuit (14 jours)",
            font=("Arial", 12),
            fg_color="#FFA500",
            hover_color="#FF8C00",
            command=self.start_trial
        )
        self.trial_button.pack(side="left")

        # Informations d'usage
        self.usage_frame = ctk.CTkFrame(self, fg_color="#161920")
        self.usage_frame.pack(fill="x", padx=20, pady=10)

        usage_title = ctk.CTkLabel(
            self.usage_frame,
            text="📈 Utilisation",
            font=("Arial", 14, "bold"),
            text_color="#00BFFF"
        )
        usage_title.pack(pady=(10, 5))

        self.usage_text = ctk.CTkLabel(
            self.usage_frame,
            text="—",
            font=("Arial", 12),
            text_color="#CCCCCC",
            justify="left"
        )
        self.usage_text.pack(pady=(0, 10), padx=10, anchor="w")

        # Boutons de contrôle bas de page
        control_frame = ctk.CTkFrame(self, fg_color="transparent")
        control_frame.pack(fill="x", padx=20, pady=20)

        self.copy_key_button = ctk.CTkButton(
            control_frame,
            text="📋 Copier la clé",
            font=("Arial", 12),
            fg_color="#3b82f6",
            hover_color="#2563eb",
            command=self.copy_current_key
        )
        self.copy_key_button.pack(side="left", padx=(0, 10))

        self.deactivate_button = ctk.CTkButton(
            control_frame,
            text="🔒 Désactiver",
            font=("Arial", 12),
            fg_color="#ef4444",
            hover_color="#dc2626",
            command=self.deactivate_license
        )
        self.deactivate_button.pack(side="left")

        # Espace flexible
        ctk.CTkLabel(control_frame, text="", width=10).pack(side="left", expand=True)

        if not self.force_activation:
            close_button = ctk.CTkButton(
                control_frame,
                text="Fermer",
                font=("Arial", 12),
                fg_color="#2a2d36",
                hover_color="#1f2430",
                command=self.on_close
            )
            close_button.pack(side="right")
        else:
            info_forced = ctk.CTkLabel(
                control_frame,
                text="Activation requise pour continuer",
                font=("Arial", 12, "bold"),
                text_color="#FFAA00"
            )
            info_forced.pack(side="right")

    # ---------- interactions ----------
    def set_status_text(self, text: str, color: str = "#CCCCCC"):
        try:
            self.status_text.configure(text=text, text_color=color)
        except Exception:
            pass

    def set_usage_text(self, text: str):
        try:
            self.usage_text.configure(text=text)
        except Exception:
            pass

    def populate_info(self, info: Dict[str, Any]):
        """Remplit le bloc d'informations détaillées"""
        def fmt_date(ts: Optional[str]) -> str:
            if not ts:
                return "-"
            try:
                # accepte isoformat ou epoch str
                if ts.isdigit():
                    dt = datetime.fromtimestamp(int(ts))
                else:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.strftime("%d/%m/%Y %H:%M")
            except Exception:
                return str(ts)

        mapping = {
            "Type": info.get("plan") or info.get("type") or "-",
            "Email": info.get("email") or "-",
            "Clé": (info.get("key_masked") or info.get("key") or "-"),
            "Statut": info.get("status") or ("Valide" if info.get("valid") else "Invalide"),
            "Activée le": fmt_date(info.get("activated_at")),
            "Expire le": fmt_date(info.get("expires_at")),
            "Jours restants": str(info.get("days_left")) if info.get("days_left") is not None else "-"
        }
        for label, widget in self.info_rows.items():
            try:
                widget.configure(text=mapping.get(label, "-"))
            except Exception:
                pass

    # ---------- actions licence ----------
    def update_license_status_async(self):
        """Rafraîchit le statut dans un thread pour ne pas bloquer l'UI"""
        self.set_status_text("Vérification en cours…", "#AAAAAA")

        def worker():
            try:
                status = self.safe_get_status()
                usage = self.safe_get_usage()

                def update():
                    color = "#00FF88" if (status.get("valid") is True or status.get("status") == "valid") else "#FF5555"
                    human = self.humanize_status(status)
                    self.set_status_text(human, color)
                    self.populate_info(status)
                    self.set_usage_text(self.humanize_usage(usage))
                    # état boutons
                    self.update_buttons_enabled(status)
                self.after(0, update)
            except Exception as e:
                self.logger.log_error(e, "license_status", "Erreur lors de la récupération du statut licence")
                self.after(0, lambda: self.set_status_text("Erreur de récupération du statut", "#FF5555"))

        threading.Thread(target=worker, daemon=True).start()

    def update_buttons_enabled(self, status: Dict[str, Any]):
        valid = bool(status.get("valid") or status.get("status") == "valid")
        try:
            # Activer/désactiver actions selon état
            self.deactivate_button.configure(state=("normal" if valid else "disabled"))
            # Pendant activation forcée, empêcher de fermer
            if self.force_activation:
                # On autorise quand même essais/activation
                pass
        except Exception:
            pass

    def activate_license(self):
        """Active une licence avec email + clé"""
        email = self.email_entry.get().strip()
        key = self.license_key_entry.get().strip()

        if not email or "@" not in email:
            messagebox.showwarning("Validation", "Veuillez saisir un email valide.")
            return
        if not key or len(key) < 10:
            messagebox.showwarning("Validation", "Veuillez saisir une clé de licence valide.")
            return

        self.disable_inputs(True)
        self.set_status_text("Activation en cours…", "#AAAAAA")

        def worker():
            ok = False
            msg = ""
            try:
                if hasattr(self.license_manager, "activate"):
                    resp = self.license_manager.activate(email=email, key=key)
                    ok = bool(resp.get("success", False))
                    msg = resp.get("message", "")
                else:
                    msg = "Fonction d'activation indisponible."
            except Exception as e:
                self.logger.log_error(e, "license_activate", "Erreur activation licence")
                msg = str(e)

            def done():
                self.disable_inputs(False)
                if ok:
                    messagebox.showinfo("Licence", "Activation réussie ✅")
                    self.update_license_status_async()
                    if self.force_activation:
                        self.result = {"activated": True}
                        self.on_close()
                else:
                    messagebox.showerror("Licence", f"Échec de l'activation.\n{msg or ''}")
                    self.set_status_text("Activation échouée", "#FF5555")

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def start_trial(self):
        """Démarre un essai gratuit (si supporté)"""
        email = self.email_entry.get().strip()
        if not email or "@" not in email:
            messagebox.showwarning("Validation", "Veuillez saisir un email valide pour démarrer l'essai.")
            return

        self.disable_inputs(True)
        self.set_status_text("Activation de l'essai en cours…", "#AAAAAA")

        def worker():
            ok = False
            msg = ""
            try:
                if hasattr(self.license_manager, "start_trial"):
                    resp = self.license_manager.start_trial(email=email)
                    ok = bool(resp.get("success", False))
                    msg = resp.get("message", "")
                else:
                    msg = "Essai gratuit non disponible."
            except Exception as e:
                self.logger.log_error(e, "license_trial", "Erreur démarrage essai")
                msg = str(e)

            def done():
                self.disable_inputs(False)
                if ok:
                    messagebox.showinfo("Essai", "Essai gratuit activé ✅")
                    self.update_license_status_async()
                    if self.force_activation:
                        self.result = {"trial": True}
                        self.on_close()
                else:
                    messagebox.showerror("Essai", f"Échec de l'activation de l'essai.\n{msg or ''}")
                    self.set_status_text("Essai non activé", "#FFAA00")

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def deactivate_license(self):
        """Désactive la licence active (si supporté)"""
        if not messagebox.askyesno("Désactivation", "Voulez-vous vraiment désactiver la licence sur cet appareil ?"):
            return

        self.disable_inputs(True)
        self.set_status_text("Désactivation…", "#AAAAAA")

        def worker():
            ok = False
            msg = ""
            try:
                if hasattr(self.license_manager, "deactivate"):
                    resp = self.license_manager.deactivate()
                    ok = bool(resp.get("success", False))
                    msg = resp.get("message", "")
                else:
                    msg = "Désactivation non disponible."
            except Exception as e:
                self.logger.log_error(e, "license_deactivate", "Erreur désactivation licence")
                msg = str(e)

            def done():
                self.disable_inputs(False)
                if ok:
                    messagebox.showinfo("Licence", "Licence désactivée ✅")
                    self.update_license_status_async()
                else:
                    messagebox.showerror("Licence", f"Échec de la désactivation.\n{msg or ''}")
                    self.set_status_text("Désactivation échouée", "#FF5555")

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def copy_current_key(self):
        """Copie la clé affichée dans le presse-papiers"""
        try:
            key_text = self.info_rows.get("Clé").cget("text")
            if not key_text or key_text == "-":
                messagebox.showinfo("Copie", "Aucune clé à copier.")
                return
            self.clipboard_clear()
            self.clipboard_append(key_text)
            messagebox.showinfo("Copie", "Clé copiée dans le presse-papiers.")
        except Exception as e:
            self.logger.log_error(e, "license_copy_key", "Erreur copie clé")
            messagebox.showerror("Copie", "Impossible de copier la clé.")

    # ---------- utilitaires backend ----------
    def safe_get_status(self) -> Dict[str, Any]:
        """Récupère un statut unifié, quels que soient les méthodes dispo du manager"""
        status: Dict[str, Any] = {}
        try:
            if hasattr(self.license_manager, "get_status"):
                status = dict(self.license_manager.get_status() or {})
            else:
                # fallback: construire depuis is_valid, get_license_info, etc.
                valid = False
                if hasattr(self.license_manager, "is_valid"):
                    valid = bool(self.license_manager.is_valid())
                info = {}
                if hasattr(self.license_manager, "get_license_info"):
                    info = dict(self.license_manager.get_license_info() or {})
                status.update(info)
                status.setdefault("valid", valid)
                status.setdefault("status", "valid" if valid else "invalid")
        except Exception as e:
            self.logger.log_error(e, "license_status_read", "Erreur lecture statut")
            status.setdefault("valid", False)
            status.setdefault("status", "error")
        return status

    def safe_get_usage(self) -> Dict[str, Any]:
        """Récupère l'usage (limites, compteurs) si disponible"""
        usage: Dict[str, Any] = {}
        try:
            if hasattr(self.license_manager, "get_usage"):
                usage = dict(self.license_manager.get_usage() or {})
        except Exception as e:
            self.logger.log_error(e, "license_usage_read", "Erreur lecture usage")
        return usage

    @staticmethod
    def humanize_status(status: Dict[str, Any]) -> str:
        st = (status.get("status") or "").lower()
        valid = bool(status.get("valid")) or st == "valid"
        plan = status.get("plan") or status.get("type") or "—"
        email = status.get("email") or "—"
        if valid:
            return f"Licence valide ({plan}) pour {email}"
        if st == "trial":
            return f"Essai actif ({plan}) pour {email}"
        if st == "expired":
            return "Licence expirée"
        if st == "error":
            return "Erreur lors de la vérification"
        return "Licence invalide"

    @staticmethod
    def humanize_usage(usage: Dict[str, Any]) -> str:
        if not usage:
            return "Aucune donnée d'utilisation disponible."
        parts = []
        for k, v in usage.items():
            parts.append(f"• {k}: {v}")
        return "\n".join(parts)

    def disable_inputs(self, disable: bool):
        state = "disabled" if disable else "normal"
        try:
            self.email_entry.configure(state=state)
            self.license_key_entry.configure(state=state)
            self.activate_button.configure(state=state)
            self.trial_button.configure(state=state)
            self.deactivate_button.configure(state=state)
            self.copy_key_button.configure(state=state)
        except Exception:
            pass

    # ---------- fermeture ----------
    def on_close(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def on_forced_close_block(self):
        messagebox.showwarning(
            "Activation requise",
            "Vous devez activer une licence ou démarrer un essai pour continuer."
        )


# =========================
#   Aide à l’intégration
# =========================
def show_license_dialog(parent, force_activation: bool = False) -> Optional[Dict[str, Any]]:
    """
    Ouvre la boîte de dialogue licence et retourne le résultat éventuel :
    - {"activated": True} après activation
    - {"trial": True} après démarrage d'essai
    - None si fermeture sans action
    """
    dlg = LicenseDialog(parent, force_activation=force_activation)
    parent.wait_window(dlg)
    return dlg.result


__all__ = ["LicenseDialog", "show_license_dialog"]
