"""
Keyboard Shortcuts Manager - Gestionnaire de raccourcis clavier
"""
import customtkinter as ctk
import sys
from typing import Dict, Callable, Optional, List, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


def is_mac() -> bool:
    """Vérifie si on est sur Mac"""
    return sys.platform == "darwin"


def get_modifier() -> str:
    """Retourne le modificateur de touche selon la plateforme (Command sur Mac, Control ailleurs)"""
    return "Command" if is_mac() else "Control"


@dataclass
class Shortcut:
    """Représente un raccourci clavier"""
    key: str  # ex: "Control-s", "Alt-n"
    name: str  # Nom descriptif
    description: str
    category: str
    callback: Optional[Callable] = None
    is_global: bool = False  # Global ou local au widget


class KeyboardShortcutsManager:
    """Gestionnaire centralisé des raccourcis clavier"""

    DEFAULT_SHORTCUTS = {
        # Navigation
        "Control-1": ("Accueil", "Aller à l'accueil", "Navigation"),
        "Control-2": ("Watchlist", "Ouvrir la watchlist", "Navigation"),
        "Control-3": ("Portfolio", "Ouvrir le portfolio", "Navigation"),
        "Control-4": ("Analyse", "Ouvrir l'analyse", "Navigation"),
        "Control-5": ("Formation", "Ouvrir la formation", "Navigation"),
        "Control-6": ("Communauté", "Ouvrir la communauté", "Navigation"),

        # Actions
        "Control-n": ("Nouvelle recherche", "Nouvelle recherche d'action", "Actions"),
        "Control-s": ("Sauvegarder", "Sauvegarder la configuration", "Actions"),
        "Control-r": ("Actualiser", "Rafraîchir les données", "Actions"),
        "Control-e": ("Exporter", "Exporter les données", "Actions"),
        "Control-p": ("Imprimer", "Imprimer le rapport", "Actions"),

        # Interface
        "Control-t": ("Thème", "Changer le thème", "Interface"),
        "Control-plus": ("Zoom +", "Augmenter le zoom", "Interface"),
        "Control-minus": ("Zoom -", "Diminuer le zoom", "Interface"),
        "Control-0": ("Zoom reset", "Réinitialiser le zoom", "Interface"),
        "F11": ("Plein écran", "Mode plein écran", "Interface"),
        "Escape": ("Fermer popup", "Fermer le popup actuel", "Interface"),

        # Trading
        "Control-b": ("Acheter", "Ordre d'achat", "Trading"),
        "Control-Shift-b": ("Vendre", "Ordre de vente", "Trading"),
        "Control-a": ("Alerte", "Créer une alerte", "Trading"),
        "Control-w": ("Ajouter watchlist", "Ajouter à la watchlist", "Trading"),

        # Analyse
        "Control-i": ("Indicateurs", "Ouvrir les indicateurs", "Analyse"),
        "Control-d": ("Dessiner", "Mode dessin", "Analyse"),
        "Control-m": ("ML Prédiction", "Lancer prédiction ML", "Analyse"),

        # Général
        "Control-q": ("Quitter", "Fermer l'application", "Général"),
        "Control-comma": ("Préférences", "Ouvrir les préférences", "Général"),
        "F1": ("Aide", "Ouvrir l'aide", "Général"),
        "Control-Shift-p": ("Palette", "Ouvrir la palette de commandes", "Général"),
    }

    def __init__(self, root: ctk.CTk = None):
        self.root = root
        self.shortcuts: Dict[str, Shortcut] = {}
        self.callbacks: Dict[str, Callable] = {}
        self.enabled = True

        self.config_path = Path.home() / ".helixone" / "shortcuts.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_default_shortcuts()
        self._load_custom_shortcuts()

    def _init_default_shortcuts(self):
        """Initialise les raccourcis par défaut"""
        for key, (name, desc, category) in self.DEFAULT_SHORTCUTS.items():
            self.shortcuts[key] = Shortcut(
                key=key,
                name=name,
                description=desc,
                category=category
            )

    def _load_custom_shortcuts(self):
        """Charge les raccourcis personnalisés"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    custom = json.load(f)
                    # Mettre à jour les touches personnalisées
                    for old_key, new_key in custom.get("remapped", {}).items():
                        if old_key in self.shortcuts:
                            shortcut = self.shortcuts.pop(old_key)
                            shortcut.key = new_key
                            self.shortcuts[new_key] = shortcut
        except Exception:
            pass

    def _save_custom_shortcuts(self):
        """Sauvegarde les raccourcis personnalisés"""
        try:
            remapped = {}
            for key, shortcut in self.shortcuts.items():
                default_key = None
                for dk, (name, _, _) in self.DEFAULT_SHORTCUTS.items():
                    if name == shortcut.name and dk != key:
                        default_key = dk
                        break
                if default_key:
                    remapped[default_key] = key

            with open(self.config_path, 'w') as f:
                json.dump({"remapped": remapped}, f)
        except Exception:
            pass

    def _platform_key(self, key: str) -> str:
        """Convertit un raccourci en format plateforme (Control -> Command sur Mac)"""
        if is_mac() and "Control" in key:
            return key.replace("Control", "Command")
        return key

    def bind(self, key: str, callback: Callable):
        """Associe un callback à un raccourci"""
        if key in self.shortcuts:
            self.shortcuts[key].callback = callback
            self.callbacks[key] = callback

            if self.root:
                # Convertir Control en Command sur Mac
                binding_key = self._platform_key(key)
                self.root.bind(f"<{binding_key}>", lambda e: self._execute(key))

    def bind_widget(self, widget: ctk.CTkBaseClass, key: str, callback: Callable):
        """Associe un callback à un raccourci pour un widget spécifique"""
        binding_key = self._platform_key(key)
        widget.bind(f"<{binding_key}>", lambda e: callback() if self.enabled else None)

    def unbind(self, key: str):
        """Supprime l'association d'un raccourci"""
        if key in self.callbacks:
            del self.callbacks[key]
        if self.root:
            try:
                binding_key = self._platform_key(key)
                self.root.unbind(f"<{binding_key}>")
            except Exception:
                pass

    def _execute(self, key: str):
        """Exécute le callback associé à un raccourci"""
        if not self.enabled:
            return

        if key in self.callbacks:
            try:
                self.callbacks[key]()
            except Exception as e:
                print(f"Erreur exécution raccourci {key}: {e}")

    def remap(self, old_key: str, new_key: str):
        """Remplace un raccourci par un autre"""
        if old_key in self.shortcuts:
            shortcut = self.shortcuts.pop(old_key)

            # Unbind old
            if old_key in self.callbacks:
                callback = self.callbacks.pop(old_key)
                if self.root:
                    try:
                        old_binding = self._platform_key(old_key)
                        self.root.unbind(f"<{old_binding}>")
                    except Exception:
                        pass

                # Bind new
                shortcut.key = new_key
                self.shortcuts[new_key] = shortcut
                self.callbacks[new_key] = callback
                if self.root:
                    new_binding = self._platform_key(new_key)
                    self.root.bind(f"<{new_binding}>", lambda e: self._execute(new_key))

            self._save_custom_shortcuts()

    def reset_to_default(self):
        """Réinitialise tous les raccourcis par défaut"""
        # Unbind all
        for key in list(self.shortcuts.keys()):
            if self.root:
                try:
                    binding_key = self._platform_key(key)
                    self.root.unbind(f"<{binding_key}>")
                except Exception:
                    pass

        self.shortcuts.clear()
        self.callbacks.clear()
        self._init_default_shortcuts()

        # Delete custom config
        if self.config_path.exists():
            self.config_path.unlink()

    def get_shortcuts_by_category(self) -> Dict[str, List[Shortcut]]:
        """Retourne les raccourcis groupés par catégorie"""
        categories = {}
        for shortcut in self.shortcuts.values():
            if shortcut.category not in categories:
                categories[shortcut.category] = []
            categories[shortcut.category].append(shortcut)
        return categories

    def enable(self):
        """Active les raccourcis"""
        self.enabled = True

    def disable(self):
        """Désactive les raccourcis (utile pendant la saisie)"""
        self.enabled = False

    def format_key(self, key: str) -> str:
        """Formate un raccourci pour l'affichage"""
        parts = key.split("-")
        formatted = []

        for part in parts:
            if part == "Control" or part == "Command":
                formatted.append("⌘" if is_mac() else "Ctrl")
            elif part == "Alt":
                formatted.append("⌥" if is_mac() else "Alt")
            elif part == "Shift":
                formatted.append("⇧" if is_mac() else "Shift")
            elif part == "plus":
                formatted.append("+")
            elif part == "minus":
                formatted.append("-")
            else:
                formatted.append(part.upper() if len(part) == 1 else part)

        return " + ".join(formatted)


# Instance globale
_shortcuts_manager = None


def get_shortcuts_manager(root: ctk.CTk = None) -> KeyboardShortcutsManager:
    """Retourne l'instance globale du gestionnaire de raccourcis"""
    global _shortcuts_manager
    if _shortcuts_manager is None:
        _shortcuts_manager = KeyboardShortcutsManager(root)
    elif root is not None and _shortcuts_manager.root is None:
        _shortcuts_manager.root = root
    return _shortcuts_manager


class ShortcutsPanel(ctk.CTkFrame):
    """Panneau d'affichage et configuration des raccourcis"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color="transparent")

        self.manager = get_shortcuts_manager()
        self.editing_key: Optional[str] = None

        self._create_ui()

    def _create_ui(self):
        """Crée l'interface"""
        # Header
        header = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=10)
        header.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            header,
            text="⌨️ Raccourcis Clavier",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=15)

        # Toolbar
        toolbar = ctk.CTkFrame(header, fg_color="transparent")
        toolbar.pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkButton(
            toolbar,
            text="🔄 Réinitialiser",
            font=ctk.CTkFont(size=12),
            width=120,
            height=30,
            fg_color="#553333",
            hover_color="#773333",
            command=self._reset_shortcuts
        ).pack(side="left")

        # Search
        self.search_var = ctk.StringVar()
        search_entry = ctk.CTkEntry(
            toolbar,
            placeholder_text="🔍 Rechercher...",
            textvariable=self.search_var,
            width=200,
            height=30
        )
        search_entry.pack(side="right")
        self.search_var.trace_add("write", lambda *args: self._refresh_list())

        # Shortcuts list
        self.list_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self._refresh_list()

    def _refresh_list(self):
        """Actualise la liste des raccourcis"""
        for widget in self.list_frame.winfo_children():
            widget.destroy()

        search = self.search_var.get().lower()
        categories = self.manager.get_shortcuts_by_category()

        for category, shortcuts in categories.items():
            # Filtrer
            filtered = [s for s in shortcuts
                       if search in s.name.lower() or search in s.description.lower()]

            if not filtered:
                continue

            # Header de catégorie
            cat_frame = ctk.CTkFrame(self.list_frame, fg_color="#252540", corner_radius=10)
            cat_frame.pack(fill="x", padx=5, pady=5)

            ctk.CTkLabel(
                cat_frame,
                text=f"📁 {category}",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#00d4ff"
            ).pack(anchor="w", padx=15, pady=10)

            # Raccourcis
            for shortcut in filtered:
                self._create_shortcut_row(cat_frame, shortcut)

    def _create_shortcut_row(self, parent, shortcut: Shortcut):
        """Crée une ligne de raccourci"""
        row = ctk.CTkFrame(parent, fg_color="#1e1e32", corner_radius=5)
        row.pack(fill="x", padx=10, pady=2)

        # Info
        info = ctk.CTkFrame(row, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True, padx=10, pady=8)

        ctk.CTkLabel(
            info,
            text=shortcut.name,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="white"
        ).pack(anchor="w")

        ctk.CTkLabel(
            info,
            text=shortcut.description,
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        ).pack(anchor="w")

        # Key display
        key_text = self.manager.format_key(shortcut.key)

        key_btn = ctk.CTkButton(
            row,
            text=key_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            width=120,
            height=35,
            fg_color="#333355",
            hover_color="#444466",
            command=lambda s=shortcut: self._start_edit(s)
        )
        key_btn.pack(side="right", padx=10, pady=5)

    def _start_edit(self, shortcut: Shortcut):
        """Démarre l'édition d'un raccourci"""
        self.editing_key = shortcut.key

        dialog = ctk.CTkToplevel(self)
        dialog.title("Modifier le raccourci")
        dialog.geometry("400x200")
        dialog.resizable(False, False)
        dialog.attributes("-topmost", True)

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 400) // 2
        y = (dialog.winfo_screenheight() - 200) // 2
        dialog.geometry(f"+{x}+{y}")

        content = ctk.CTkFrame(dialog, fg_color="#1a1a2e")
        content.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(
            content,
            text=f"Nouveau raccourci pour:\n{shortcut.name}",
            font=ctk.CTkFont(size=14),
            text_color="white"
        ).pack(pady=(30, 10))

        self.new_key_label = ctk.CTkLabel(
            content,
            text="Appuyez sur une touche...",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00d4ff"
        )
        self.new_key_label.pack(pady=20)

        self.captured_key = None

        def on_key(event):
            parts = []
            if event.state & 0x4:  # Control
                parts.append("Control")
            if event.state & 0x1:  # Shift
                parts.append("Shift")
            if event.state & 0x8:  # Alt
                parts.append("Alt")

            key = event.keysym
            if key not in ["Control_L", "Control_R", "Shift_L", "Shift_R", "Alt_L", "Alt_R"]:
                parts.append(key)
                self.captured_key = "-".join(parts)
                self.new_key_label.configure(text=self.manager.format_key(self.captured_key))

        dialog.bind("<Key>", on_key)

        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkButton(
            btn_frame,
            text="Annuler",
            width=100,
            fg_color="#553333",
            command=dialog.destroy
        ).pack(side="left", padx=5)

        def save():
            if self.captured_key and self.editing_key:
                self.manager.remap(self.editing_key, self.captured_key)
                self._refresh_list()
            dialog.destroy()

        ctk.CTkButton(
            btn_frame,
            text="Sauvegarder",
            width=100,
            fg_color="#00d4ff",
            command=save
        ).pack(side="right", padx=5)

    def _reset_shortcuts(self):
        """Réinitialise les raccourcis"""
        self.manager.reset_to_default()
        self._refresh_list()


class ShortcutsHelpPopup(ctk.CTkToplevel):
    """Popup d'aide des raccourcis (F1)"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.title("Raccourcis Clavier")
        self.geometry("500x600")
        self.resizable(True, True)

        self.manager = get_shortcuts_manager()

        self._create_ui()

        # Position
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 500) // 2
        y = (self.winfo_screenheight() - 600) // 2
        self.geometry(f"+{x}+{y}")

    def _create_ui(self):
        """Crée l'interface"""
        content = ctk.CTkFrame(self, fg_color="#1a1a2e")
        content.pack(fill="both", expand=True)

        # Header
        header = ctk.CTkFrame(content, fg_color="#252540")
        header.pack(fill="x")

        ctk.CTkLabel(
            header,
            text="⌨️ Raccourcis Clavier",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=15)

        # List
        list_frame = ctk.CTkScrollableFrame(content, fg_color="transparent")
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        categories = self.manager.get_shortcuts_by_category()

        for category, shortcuts in categories.items():
            # Category header
            ctk.CTkLabel(
                list_frame,
                text=category,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#00d4ff"
            ).pack(anchor="w", padx=10, pady=(15, 5))

            for shortcut in shortcuts:
                row = ctk.CTkFrame(list_frame, fg_color="#1e1e32", corner_radius=5)
                row.pack(fill="x", padx=5, pady=2)

                ctk.CTkLabel(
                    row,
                    text=shortcut.name,
                    font=ctk.CTkFont(size=12),
                    text_color="white"
                ).pack(side="left", padx=10, pady=8)

                key_text = self.manager.format_key(shortcut.key)
                ctk.CTkLabel(
                    row,
                    text=key_text,
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#888888",
                    fg_color="#333355",
                    corner_radius=5,
                    padx=10,
                    pady=3
                ).pack(side="right", padx=10, pady=5)

        # Close button
        ctk.CTkButton(
            content,
            text="Fermer (Echap)",
            font=ctk.CTkFont(size=13),
            height=40,
            fg_color="#333355",
            command=self.destroy
        ).pack(fill="x", padx=20, pady=15)

        self.bind("<Escape>", lambda e: self.destroy())


class CommandPalette(ctk.CTkToplevel):
    """Palette de commandes style VS Code (Ctrl+Shift+P)"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.title("")
        self.overrideredirect(True)  # Pas de décorations
        self.attributes("-topmost", True)

        self.manager = get_shortcuts_manager()
        self.commands = self._build_commands()
        self.filtered_commands = self.commands.copy()

        self._create_ui()

        # Position centrée en haut
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 500) // 2
        y = self.winfo_screenheight() // 4
        self.geometry(f"500x400+{x}+{y}")

        self.focus_force()
        self.search_entry.focus()

    def _build_commands(self) -> List[Dict]:
        """Construit la liste des commandes"""
        commands = []
        for key, shortcut in self.manager.shortcuts.items():
            commands.append({
                "name": shortcut.name,
                "description": shortcut.description,
                "key": key,
                "category": shortcut.category,
                "callback": shortcut.callback
            })
        return commands

    def _create_ui(self):
        """Crée l'interface"""
        content = ctk.CTkFrame(self, fg_color="#1a1a2e", border_width=2, border_color="#333355")
        content.pack(fill="both", expand=True)

        # Search
        search_frame = ctk.CTkFrame(content, fg_color="#252540")
        search_frame.pack(fill="x")

        self.search_var = ctk.StringVar()
        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Tapez une commande...",
            textvariable=self.search_var,
            font=ctk.CTkFont(size=16),
            height=50,
            border_width=0,
            fg_color="transparent"
        )
        self.search_entry.pack(fill="x", padx=15, pady=10)
        self.search_var.trace_add("write", lambda *args: self._filter())

        # Results
        self.results_frame = ctk.CTkScrollableFrame(content, fg_color="transparent")
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self._refresh_results()

        # Bindings
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: self._execute_selected())
        self.bind("<Up>", lambda e: self._move_selection(-1))
        self.bind("<Down>", lambda e: self._move_selection(1))

        self.selected_index = 0

    def _filter(self):
        """Filtre les commandes"""
        search = self.search_var.get().lower()

        if not search:
            self.filtered_commands = self.commands.copy()
        else:
            self.filtered_commands = [
                c for c in self.commands
                if search in c["name"].lower() or search in c["description"].lower()
            ]

        self.selected_index = 0
        self._refresh_results()

    def _refresh_results(self):
        """Actualise les résultats"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        for i, cmd in enumerate(self.filtered_commands[:10]):
            is_selected = i == self.selected_index
            bg_color = "#333355" if is_selected else "#1e1e32"

            row = ctk.CTkFrame(self.results_frame, fg_color=bg_color, corner_radius=5, cursor="hand2")
            row.pack(fill="x", pady=2)

            # Bind click
            row.bind("<Button-1>", lambda e, idx=i: self._select_and_execute(idx))

            info = ctk.CTkFrame(row, fg_color="transparent")
            info.pack(side="left", fill="x", expand=True, padx=10, pady=8)

            ctk.CTkLabel(
                info,
                text=cmd["name"],
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="white"
            ).pack(anchor="w")

            ctk.CTkLabel(
                info,
                text=cmd["description"],
                font=ctk.CTkFont(size=11),
                text_color="#888888"
            ).pack(anchor="w")

            # Key
            key_text = self.manager.format_key(cmd["key"])
            ctk.CTkLabel(
                row,
                text=key_text,
                font=ctk.CTkFont(size=10),
                text_color="#666666"
            ).pack(side="right", padx=10)

    def _move_selection(self, delta: int):
        """Déplace la sélection"""
        self.selected_index = max(0, min(len(self.filtered_commands) - 1, self.selected_index + delta))
        self._refresh_results()

    def _select_and_execute(self, index: int):
        """Sélectionne et exécute"""
        self.selected_index = index
        self._execute_selected()

    def _execute_selected(self):
        """Exécute la commande sélectionnée"""
        if 0 <= self.selected_index < len(self.filtered_commands):
            cmd = self.filtered_commands[self.selected_index]
            self.destroy()
            if cmd["callback"]:
                try:
                    cmd["callback"]()
                except Exception as e:
                    print(f"Erreur exécution commande: {e}")
