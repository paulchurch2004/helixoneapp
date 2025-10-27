import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import threading
import time
from datetime import datetime
from PIL import Image
from pathlib import Path
import mimetypes
import hashlib
import shutil
import re
import sqlite3
import uuid
from typing import Dict, List, Optional, Any


class CommunityChat(ctk.CTkFrame):
    """Système de chat communautaire pour HelixOne"""

    def __init__(self, parent, user_profile):
        super().__init__(parent, fg_color="#0a0e12")
        self.user_profile = user_profile or {"username": "Utilisateur"}
        self.current_channel = "general"
        self.messages = {}
        self.channels = {}

        # Configuration des dossiers
        self.setup_directories()

        # Charger les données
        self.load_channels()
        self.load_messages()

        # Interface
        self.setup_ui()

        # Démarrer le système de mise à jour
        self.start_message_polling()

    # -----------------------------
    # Fichiers & dossiers
    # -----------------------------
    def setup_directories(self):
        """Crée les dossiers nécessaires"""
        self.data_dir = Path("data/community")
        self.messages_dir = self.data_dir / "messages"
        self.uploads_dir = self.data_dir / "uploads"
        self.channels_file = self.data_dir / "channels.json"

        for directory in [self.data_dir, self.messages_dir, self.uploads_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # UI
    # -----------------------------
    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        # Container principal
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Sidebar avec canaux
        self.sidebar = self.create_sidebar(main_container)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))

        # Zone de chat principale
        self.chat_area = self.create_chat_area(main_container)
        self.chat_area.pack(side="right", fill="both", expand=True)

    def create_sidebar(self, parent):
        """Crée la sidebar avec la liste des canaux"""
        sidebar = ctk.CTkFrame(parent, width=250, fg_color="#161920")
        sidebar.pack_propagate(False)

        # Header
        header = ctk.CTkFrame(sidebar, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=20)

        ctk.CTkLabel(
            header,
            text="💬 Canaux",
            font=("Arial", 18, "bold"),
            text_color="#00BFFF",
        ).pack()

        # Bouton nouveau canal
        ctk.CTkButton(
            header,
            text="+ Nouveau Canal",
            width=150,
            height=30,
            command=self.create_new_channel,
        ).pack(pady=(10, 0))

        # Liste des canaux
        self.channels_frame = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
        self.channels_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.update_channels_list()

        # Section utilisateurs connectés (statique dans la classe de base)
        self.create_users_section(sidebar)

        return sidebar

    def create_users_section(self, parent):
        """Section des utilisateurs connectés (simulation)"""
        separator = ctk.CTkFrame(parent, height=1, fg_color="#333333")
        separator.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(
            parent,
            text="👥 En ligne (3)",
            font=("Arial", 12, "bold"),
            text_color="#888888",
        ).pack(padx=15, anchor="w")

        # Liste des utilisateurs (simulation)
        users = ["TradingPro", "AlphaInvestor", "Vous"]
        for user in users:
            user_frame = ctk.CTkFrame(parent, fg_color="#2a2d36")
            user_frame.pack(fill="x", padx=15, pady=2)

            # Statut en ligne
            status_color = "#00FF88" if user != "Vous" else "#00BFFF"
            ctk.CTkLabel(
                user_frame,
                text=f"🟢 {user}",
                font=("Arial", 10),
                text_color=status_color,
            ).pack(padx=10, pady=5)

    def create_chat_area(self, parent):
        """Crée la zone de chat principale"""
        chat_container = ctk.CTkFrame(parent, fg_color="#1c1f26")

        # Header du canal
        self.chat_header = self.create_chat_header(chat_container)
        self.chat_header.pack(fill="x", padx=20, pady=(20, 10))

        # Zone des messages
        self.messages_area = ctk.CTkScrollableFrame(chat_container, fg_color="#101418")
        self.messages_area.pack(fill="both", expand=True, padx=20, pady=10)

        # Zone de saisie
        self.input_area = self.create_input_area(chat_container)
        self.input_area.pack(fill="x", padx=20, pady=(0, 20))

        # Afficher messages initiaux
        self.display_messages()

        return chat_container

    def create_chat_header(self, parent):
        """Header du canal actuel"""
        header = ctk.CTkFrame(parent, fg_color="#2a2d36")

        # Informations du canal
        info_frame = ctk.CTkFrame(header, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=20, pady=15)

        self.channel_title = ctk.CTkLabel(
            info_frame,
            text=f"# {self.current_channel}",
            font=("Arial", 18, "bold"),
            text_color="#FFFFFF",
        )
        self.channel_title.pack(anchor="w")

        self.channel_desc = ctk.CTkLabel(
            info_frame,
            text="Canal de discussion général",
            font=("Arial", 12),
            text_color="#AAAAAA",
        )
        self.channel_desc.pack(anchor="w")

        # Boutons d'action
        actions_frame = ctk.CTkFrame(header, fg_color="transparent")
        actions_frame.pack(side="right", padx=20, pady=15)

        ctk.CTkButton(
            actions_frame, text="📌", width=40, height=30, command=self.pin_messages
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            actions_frame, text="🔍", width=40, height=30, command=self.search_messages
        ).pack(side="left", padx=2)

        return header

    def create_input_area(self, parent):
        """Zone de saisie des messages"""
        input_container = ctk.CTkFrame(parent, fg_color="#2a2d36")

        # Zone de prévisualisation des fichiers (réservée pour extensions futures)
        self.preview_frame = ctk.CTkFrame(input_container, fg_color="#1c1f26")
        # (non packé par défaut)

        # Barre d'outils
        toolbar = ctk.CTkFrame(input_container, fg_color="transparent")
        toolbar.pack(fill="x", padx=15, pady=(15, 5))

        # Boutons d'attachement
        attach_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        attach_frame.pack(side="left")

        ctk.CTkButton(
            attach_frame,
            text="📎",
            width=35,
            height=30,
            command=self.attach_file,
            fg_color="#444444",
            hover_color="#555555",
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            attach_frame,
            text="🖼️",
            width=35,
            height=30,
            command=self.attach_image,
            fg_color="#444444",
            hover_color="#555555",
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            attach_frame,
            text="📊",
            width=35,
            height=30,
            command=self.share_analysis,
            fg_color="#444444",
            hover_color="#555555",
        ).pack(side="left", padx=2)

        # Champ de texte
        text_frame = ctk.CTkFrame(input_container, fg_color="transparent")
        text_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.message_entry = ctk.CTkEntry(
            text_frame,
            placeholder_text="Tapez votre message...",
            height=40,
            font=("Arial", 14),
        )
        self.message_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.message_entry.bind("<Return>", self.send_message)
        self.message_entry.bind("<Shift-Return>", self.insert_newline)

        # Bouton d'envoi
        self.send_button = ctk.CTkButton(
            text_frame,
            text="📤",
            width=50,
            height=40,
            command=self.send_message,
            fg_color="#00BFFF",
            hover_color="#0090FF",
        )
        self.send_button.pack(side="right")

        return input_container

    # -----------------------------
    # Données
    # -----------------------------
    def load_channels(self):
        """Charge la liste des canaux"""
        try:
            if self.channels_file.exists():
                with open(self.channels_file, "r", encoding="utf-8") as f:
                    self.channels = json.load(f)
            else:
                # Canaux par défaut
                now = datetime.now().isoformat()
                self.channels = {
                    "general": {
                        "name": "Général",
                        "description": "Discussion générale",
                        "created_by": "system",
                        "created_at": now,
                    },
                    "analyses": {
                        "name": "Analyses",
                        "description": "Partage d'analyses techniques",
                        "created_by": "system",
                        "created_at": now,
                    },
                    "strategies": {
                        "name": "Stratégies",
                        "description": "Discussion sur les stratégies de trading",
                        "created_by": "system",
                        "created_at": now,
                    },
                    "aide": {
                        "name": "Aide",
                        "description": "Entraide entre traders",
                        "created_by": "system",
                        "created_at": now,
                    },
                }
                self.save_channels()
        except Exception as e:
            print(f"Erreur chargement canaux: {e}")
            self.channels = {}

    def save_channels(self):
        """Sauvegarde la liste des canaux"""
        try:
            with open(self.channels_file, "w", encoding="utf-8") as f:
                json.dump(self.channels, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erreur sauvegarde canaux: {e}")

    def load_messages(self):
        """Charge les messages de tous les canaux"""
        self.messages = {}
        for channel_id in self.channels.keys():
            messages_file = self.messages_dir / f"{channel_id}.json"
            try:
                if messages_file.exists():
                    with open(messages_file, "r", encoding="utf-8") as f:
                        self.messages[channel_id] = json.load(f)
                else:
                    self.messages[channel_id] = []
            except Exception as e:
                print(f"Erreur chargement messages {channel_id}: {e}")
                self.messages[channel_id] = []

    def save_messages(self, channel_id):
        """Sauvegarde les messages d'un canal"""
        try:
            messages_file = self.messages_dir / f"{channel_id}.json"
            with open(messages_file, "w", encoding="utf-8") as f:
                json.dump(self.messages.get(channel_id, []), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erreur sauvegarde messages {channel_id}: {e}")

    # -----------------------------
    # Liste des canaux
    # -----------------------------
    def update_channels_list(self):
        """Met à jour la liste des canaux dans la sidebar"""
        # Nettoyer la liste existante
        for widget in self.channels_frame.winfo_children():
            widget.destroy()

        for channel_id, channel_data in self.channels.items():
            self.create_channel_button(channel_id, channel_data)

    def create_channel_button(self, channel_id, channel_data):
        """Crée un bouton pour un canal"""
        # Compter les messages non lus (simulation)
        unread_count = len(self.messages.get(channel_id, [])) % 3

        btn_frame = ctk.CTkFrame(
            self.channels_frame,
            fg_color="#2a2d36" if channel_id == self.current_channel else "transparent",
        )
        btn_frame.pack(fill="x", pady=2)

        # Bouton principal du canal
        channel_btn = ctk.CTkButton(
            btn_frame,
            text=f"# {channel_data['name']}",
            font=("Arial", 12),
            height=35,
            fg_color="transparent",
            text_color="#FFFFFF" if channel_id == self.current_channel else "#CCCCCC",
            hover_color="#444444",
            anchor="w",
            command=lambda: self.switch_channel(channel_id),
        )
        channel_btn.pack(side="left", fill="x", expand=True, padx=5, pady=3)

        # Badge de messages non lus
        if unread_count > 0:
            badge = ctk.CTkLabel(
                btn_frame,
                text=str(unread_count),
                font=("Arial", 10, "bold"),
                text_color="#FFFFFF",
                fg_color="#FF4444",
                width=20,
                height=20,
                corner_radius=10,
            )
            badge.pack(side="right", padx=5)

    def switch_channel(self, channel_id):
        """Change de canal"""
        self.current_channel = channel_id
        self.update_channels_list()
        self.update_chat_header()
        self.display_messages()

    def update_chat_header(self):
        """Met à jour le header du chat"""
        if self.current_channel in self.channels:
            channel_data = self.channels[self.current_channel]
            self.channel_title.configure(text=f"# {channel_data['name']}")
            self.channel_desc.configure(text=channel_data['description'])

    # -----------------------------
    # Messages
    # -----------------------------
    def display_messages(self):
        """Affiche les messages du canal actuel (JSON de base)"""
        # Nettoyer la zone des messages
        for widget in self.messages_area.winfo_children():
            widget.destroy()

        # Afficher les messages
        channel_messages = self.messages.get(self.current_channel, [])
        for message in channel_messages[-50:]:  # Limiter à 50 messages récents
            self.create_message_widget(message)

        # Scroll vers le bas
        try:
            self.messages_area._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass
    
    def create_message_widget(self, message):
        """Crée un widget pour un message"""
        # Container du message
        msg_container = ctk.CTkFrame(self.messages_area, fg_color="transparent")
        msg_container.pack(fill="x", pady=5, padx=10)

        # Header du message (auteur + timestamp)
        header_frame = ctk.CTkFrame(msg_container, fg_color="transparent")
        header_frame.pack(fill="x")

        # Avatar et nom
        user_color = self.get_user_color(message.get("author", "?"))
        ctk.CTkLabel(
            header_frame,
            text=f"👤 {message.get('author','?')}",
            font=("Arial", 12, "bold"),
            text_color=user_color,
        ).pack(side="left")

        # Timestamp
        ts = message.get("timestamp", datetime.now().isoformat())
        try:
            timestamp = datetime.fromisoformat(ts).strftime("%H:%M")
        except Exception:
            timestamp = "--:--"
        ctk.CTkLabel(
            header_frame,
            text=timestamp,
            font=("Arial", 10),
            text_color="#888888",
        ).pack(side="left", padx=(10, 0))

        # Badge modérateur/admin (simulation)
        if message.get("author") in ["TradingPro", "AlphaInvestor"]:
            ctk.CTkLabel(header_frame, text="⭐", font=("Arial", 10)).pack(side="left", padx=(5, 0))

        # Contenu du message
        content_frame = ctk.CTkFrame(msg_container, fg_color="#2a2d36")
        content_frame.pack(fill="x", pady=(5, 0))

        # Texte du message
        if message.get("content"):
            content_label = ctk.CTkLabel(
                content_frame,
                text=message["content"],
                font=("Arial", 12),
                text_color="#FFFFFF",
                wraplength=500,
                justify="left",
            )
            content_label.pack(anchor="w", padx=15, pady=10)

        # Attachements
        if message.get("attachments"):
            self.display_attachments(content_frame, message["attachments"])

        # Réactions (simulation)
        if message.get("reactions"):
            self.display_reactions(content_frame, message["reactions"])

    def display_attachments(self, parent, attachments):
        """Affiche les attachements d'un message"""
        for attachment in attachments:
            attach_frame = ctk.CTkFrame(parent, fg_color="#1c1f26")
            attach_frame.pack(fill="x", padx=15, pady=5)

            if attachment.get("type") == "image":
                self.display_image_attachment(attach_frame, attachment)
            elif attachment.get("type") == "file":
                self.display_file_attachment(attach_frame, attachment)
            elif attachment.get("type") == "analysis":
                self.display_analysis_attachment(attach_frame, attachment)

    def display_image_attachment(self, parent, attachment):
        """Affiche une image attachée"""
        try:
            # Charger et redimensionner l'image
            image_path = self.uploads_dir / attachment["filename"]
            if image_path.exists():
                img = Image.open(image_path)
                img.thumbnail((300, 200))
                photo = ctk.CTkImage(light_image=img, size=img.size)

                img_label = ctk.CTkLabel(parent, image=photo, text="")
                img_label.image = photo  # garder une référence
                img_label.pack(padx=10, pady=10)

                # Nom du fichier
                ctk.CTkLabel(
                    parent,
                    text=attachment.get("original_name", attachment["filename"]),
                    font=("Arial", 10),
                    text_color="#AAAAAA",
                ).pack(padx=10, pady=(0, 10))
        except Exception as e:
            print(f"Erreur affichage image: {e}")

    def display_file_attachment(self, parent, attachment):
        """Affiche un fichier attaché"""
        file_frame = ctk.CTkFrame(parent, fg_color="transparent")
        file_frame.pack(fill="x", padx=10, pady=10)

        # Icône selon le type de fichier
        file_icon = self.get_file_icon(attachment.get("filename", ""))

        ctk.CTkLabel(
            file_frame,
            text=f"{file_icon} {attachment.get('original_name', attachment.get('filename',''))}",
            font=("Arial", 12),
            text_color="#FFFFFF",
        ).pack(side="left")

        # Taille du fichier
        file_size = attachment.get("size", 0)
        size_text = self.format_file_size(file_size)
        ctk.CTkLabel(
            file_frame,
            text=size_text,
            font=("Arial", 10),
            text_color="#AAAAAA",
        ).pack(side="left", padx=(10, 0))

        # Bouton de téléchargement
        ctk.CTkButton(
            file_frame,
            text="⬇️",
            width=30,
            height=25,
            command=lambda a=attachment: self.download_file(a),
        ).pack(side="right")

    def display_analysis_attachment(self, parent, attachment):
        """Affiche une analyse partagée"""
        analysis_frame = ctk.CTkFrame(parent, fg_color="#003366")
        analysis_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            analysis_frame,
            text=f"📊 Analyse: {attachment.get('ticker','?')}",
            font=("Arial", 14, "bold"),
            text_color="#00BFFF",
        ).pack(padx=15, pady=(10, 5))

        ctk.CTkLabel(
            analysis_frame,
            text=f"Score FXI: {attachment.get('score','-')}%",
            font=("Arial", 12),
            text_color="#FFFFFF",
        ).pack(padx=15, pady=(0, 10))

        if attachment.get("summary"):
            ctk.CTkLabel(
                analysis_frame,
                text=attachment.get("summary", ""),
                font=("Arial", 11),
                text_color="#DDDDDD",
                wraplength=520,
                justify="left",
            ).pack(padx=15, pady=(0, 12))

    def display_reactions(self, parent, reactions):
        """Affiche les réactions à un message"""
        reactions_frame = ctk.CTkFrame(parent, fg_color="transparent")
        reactions_frame.pack(fill="x", padx=15, pady=(0, 10))

        for emoji, count in reactions.items():
            reaction_btn = ctk.CTkButton(
                reactions_frame,
                text=f"{emoji} {count}",
                width=50,
                height=25,
                fg_color="#444444",
                hover_color="#555555",
                font=("Arial", 10),
            )
            reaction_btn.pack(side="left", padx=2)

    # -----------------------------
    # Actions utilisateur
    # -----------------------------
    def send_message(self, event=None):
        """Envoie un message"""
        content = self.message_entry.get().strip()
        if not content:
            return

        # Créer le message
        message = {
            "id": self.generate_message_id(),
            "author": self.user_profile.get("username", "Utilisateur"),
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "attachments": [],
            "reactions": {},
        }

        # Ajouter aux messages du canal
        if self.current_channel not in self.messages:
            self.messages[self.current_channel] = []

        self.messages[self.current_channel].append(message)

        # Sauvegarder
        self.save_messages(self.current_channel)

        # Effacer le champ de saisie
        self.message_entry.delete(0, "end")

        # Actualiser l'affichage
        self.display_messages()

    def insert_newline(self, event):
        """Insère une nouvelle ligne (Shift+Enter) – ignoré pour CTkEntry"""
        # Rien à faire : CTkEntry n'est pas multiligne
        return "break"

    def attach_file(self):
        """Attache un fichier"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner un fichier",
            filetypes=[
                ("Tous les fichiers", "*.*"),
                ("Documents", "*.pdf *.doc *.docx *.txt"),
                ("Feuilles de calcul", "*.xlsx *.xls *.csv"),
            ],
        )

        if file_path:
            self.process_file_attachment(file_path)

    def attach_image(self):
        """Attache une image"""
        image_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("Tous les fichiers", "*.*"),
            ],
        )

        if image_path:
            self.process_image_attachment(image_path)

    def share_analysis(self):
        """Partage une analyse"""
        dialog = AnalysisShareDialog(self, self.user_profile)
        self.wait_window(dialog)
        if getattr(dialog, "result", None):
            self.process_analysis_attachment(dialog.result)

    def process_file_attachment(self, file_path):
        """Traite un fichier attaché"""
        try:
            # Copier le fichier dans le dossier uploads
            source_path = Path(file_path)
            filename = f"{int(time.time())}_{source_path.name}"
            dest_path = self.uploads_dir / filename

            with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            # Créer l'attachment
            attachment = {
                "type": "file",
                "filename": filename,
                "original_name": source_path.name,
                "size": source_path.stat().st_size,
                "mime_type": mimetypes.guess_type(file_path)[0],
            }

            self.add_attachment_to_message(attachment)

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'attacher le fichier: {e}")

    def process_image_attachment(self, image_path):
        """Traite une image attachée"""
        try:
            # Redimensionner et optimiser l'image
            img = Image.open(image_path)

            # Redimensionner si trop grande
            max_size = (800, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Sauvegarder
            stem = Path(image_path).stem
            safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
            filename = f"{int(time.time())}_{safe_stem}.png"
            dest_path = self.uploads_dir / filename
            img.save(dest_path, "PNG", optimize=True)

            attachment = {
                "type": "image",
                "filename": filename,
                "original_name": Path(image_path).name,
                "size": dest_path.stat().st_size,
            }

            self.add_attachment_to_message(attachment)

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'attacher l'image: {e}")

    def process_analysis_attachment(self, analysis_data):
        """Traite une analyse partagée"""
        attachment = {
            "type": "analysis",
            "ticker": analysis_data["ticker"],
            "score": analysis_data["score"],
            "summary": analysis_data["summary"],
        }

        self.add_attachment_to_message(attachment)

    def add_attachment_to_message(self, attachment):
        """Ajoute un attachement au message en cours (envoie direct)"""
        message = {
            "id": self.generate_message_id(),
            "author": self.user_profile.get("username", "Utilisateur"),
            "content": "",
            "timestamp": datetime.now().isoformat(),
            "attachments": [attachment],
            "reactions": {},
        }

        if self.current_channel not in self.messages:
            self.messages[self.current_channel] = []

        self.messages[self.current_channel].append(message)
        self.save_messages(self.current_channel)
        self.display_messages()

    def create_new_channel(self):
        """Crée un nouveau canal"""
        dialog = NewChannelDialog(self)
        self.wait_window(dialog)
        if getattr(dialog, "result", None):
            channel_id = dialog.result["id"]
            # éviter collisions
            if channel_id in self.channels:
                messagebox.showerror("Erreur", "Cet identifiant de canal existe déjà.")
                return
            channel_data = {
                "name": dialog.result["name"],
                "description": dialog.result["description"],
                "created_by": self.user_profile.get("username", "Utilisateur"),
                "created_at": datetime.now().isoformat(),
            }

            self.channels[channel_id] = channel_data
            self.messages[channel_id] = []

            self.save_channels()
            self.save_messages(channel_id)
            self.update_channels_list()

    def pin_messages(self):
        """Épingle des messages importants (placeholder)"""
        messagebox.showinfo("Info", "Fonctionnalité d'épinglage à venir")

    def search_messages(self):
        """Recherche dans les messages"""
        SearchDialog(self)

    def start_message_polling(self):
        """Démarre la surveillance des nouveaux messages (rechargement local)"""
        def poll_messages():
            while True:
                try:
                    time.sleep(5)
                    self.load_messages()
                    if hasattr(self, "messages_area"):
                        self.after_idle(self.display_messages)
                except Exception as e:
                    print(f"Erreur polling messages: {e}")

        threading.Thread(target=poll_messages, daemon=True).start()

    # -----------------------------
    # Utilitaires
    # -----------------------------
    def generate_message_id(self):
        """Génère un ID unique pour un message"""
        raw = f"{time.time()}:{os.getpid()}:{self.user_profile.get('username','?')}:{os.urandom(4).hex()}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_user_color(self, username: str) -> str:
        """Retourne une couleur déterministe par utilisateur"""
        palette = [
            "#00BFFF",
            "#FF7F50",
            "#32CD32",
            "#FFD700",
            "#BA55D3",
            "#1E90FF",
            "#FF69B4",
        ]
        h = int(hashlib.sha1(username.encode("utf-8")).hexdigest(), 16)
        return palette[h % len(palette)]

    def get_file_icon(self, filename: str) -> str:
        ext = (filename or "").lower()
        if ext.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            return "🖼️"
        if ext.endswith((".pdf",)):
            return "📕"
        if ext.endswith((".xlsx", ".xls", ".csv")):
            return "📊"
        if ext.endswith((".doc", ".docx", ".txt", ".md")):
            return "📄"
        if ext.endswith((".zip", ".rar", ".7z")):
            return "🗜️"
        return "📎"

    def format_file_size(self, size_bytes: int) -> str:
        try:
            size = float(size_bytes)
        except Exception:
            size = 0.0
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    def download_file(self, attachment: dict):
        """Permet de sauvegarder localement un fichier/image attaché(e)"""
        try:
            src_path = self.uploads_dir / attachment.get("filename", "")
            if not src_path.exists():
                messagebox.showerror("Erreur", "Fichier introuvable sur le disque.")
                return

            default_name = attachment.get("original_name", attachment.get("filename", "export"))
            dst_path = filedialog.asksaveasfilename(
                title="Enregistrer sous",
                initialfile=default_name,
                defaultextension=Path(default_name).suffix or ".bin",
            )
            if not dst_path:
                return

            shutil.copyfile(src_path, dst_path)
            messagebox.showinfo("Succès", "Fichier téléchargé avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec du téléchargement: {e}")


# -----------------------------
# Dialogues auxiliaires
# -----------------------------
class NewChannelDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Nouveau canal")
        self.geometry("380x260")
        self.result = None
        self.resizable(False, False)
        self.grab_set()

        ctk.CTkLabel(self, text="Créer un nouveau canal", font=("Arial", 16, "bold")).pack(pady=12)

        form = ctk.CTkFrame(self)
        form.pack(fill="both", expand=True, padx=15, pady=10)

        # Nom affiché
        ctk.CTkLabel(form, text="Nom").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.name_entry = ctk.CTkEntry(form)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=8)

        # Identifiant (slug)
        ctk.CTkLabel(form, text="Identifiant (slug)").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.id_entry = ctk.CTkEntry(form, placeholder_text="ex: crypto")
        self.id_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=8)

        # Description
        ctk.CTkLabel(form, text="Description").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        self.desc_entry = ctk.CTkEntry(form)
        self.desc_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=8)

        form.grid_columnconfigure(1, weight=1)

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(pady=10)
        ctk.CTkButton(btns, text="Annuler", command=self.destroy).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Créer", fg_color="#00BFFF", command=self.on_create).pack(side="left", padx=6)

        self.after(100, self.name_entry.focus_set)

    def on_create(self):
        name = self.name_entry.get().strip() or "Sans titre"
        raw_id = self.id_entry.get().strip() or name
        slug = re.sub(r"[^a-z0-9_-]", "-", raw_id.lower())
        desc = self.desc_entry.get().strip() or ""
        if not slug:
            messagebox.showerror("Erreur", "L'identifiant du canal est requis.")
            return
        self.result = {"id": slug, "name": name, "description": desc}
        self.destroy()


class AnalysisShareDialog(ctk.CTkToplevel):
    def __init__(self, parent, user_profile):
        super().__init__(parent)
        self.title("Partager une analyse")
        self.geometry("420x320")
        self.result = None
        self.resizable(False, False)
        self.grab_set()

        ctk.CTkLabel(self, text="Partager une analyse", font=("Arial", 16, "bold")).pack(pady=12)
        form = ctk.CTkFrame(self)
        form.pack(fill="both", expand=True, padx=15, pady=10)

        # Ticker
        ctk.CTkLabel(form, text="Ticker").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.ticker_entry = ctk.CTkEntry(form, placeholder_text="ex: AAPL")
        self.ticker_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=8)

        # Score
        ctk.CTkLabel(form, text="Score FXI (%)").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.score_entry = ctk.CTkEntry(form, placeholder_text="ex: 78")
        self.score_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=8)

        # Résumé
        ctk.CTkLabel(form, text="Résumé").grid(row=2, column=0, sticky="nw", padx=8, pady=8)
        self.summary_box = ctk.CTkTextbox(form, height=120)
        self.summary_box.grid(row=2, column=1, sticky="nsew", padx=8, pady=8)

        form.grid_columnconfigure(1, weight=1)
        form.grid_rowconfigure(2, weight=1)

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(pady=10)
        ctk.CTkButton(btns, text="Annuler", command=self.destroy).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Partager", fg_color="#00BFFF", command=self.on_share).pack(side="left", padx=6)

        self.after(100, self.ticker_entry.focus_set)

    def on_share(self):
        ticker = (self.ticker_entry.get() or "").strip().upper()
        score_text = (self.score_entry.get() or "0").strip()
        try:
            score = max(0, min(100, int(float(score_text))))
        except Exception:
            score = 0
        summary = (self.summary_box.get("1.0", "end").strip())
        if not ticker:
            messagebox.showerror("Erreur", "Le ticker est requis.")
            return
        self.result = {"ticker": ticker, "score": score, "summary": summary}
        self.destroy()


class SearchDialog(ctk.CTkToplevel):
    def __init__(self, parent: CommunityChat):
        super().__init__(parent)
        self.parent = parent
        self.title("Recherche de messages")
        self.geometry("520x420")
        self.resizable(True, True)
        self.grab_set()

        ctk.CTkLabel(self, text="Rechercher dans le canal courant", font=("Arial", 14, "bold")).pack(pady=(12, 6))

        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=12, pady=(0, 8))

        self.query_entry = ctk.CTkEntry(top, placeholder_text="Mot-clé…")
        self.query_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(top, text="Rechercher", command=self.on_search).pack(side="right")

        self.results_frame = ctk.CTkScrollableFrame(self)
        self.results_frame.pack(fill="both", expand=True, padx=12, pady=12)

        self.bind("<Return>", lambda e: self.on_search())
        self.after(100, self.query_entry.focus_set)

    def on_search(self):
        query = (self.query_entry.get() or "").strip().lower()
        for w in self.results_frame.winfo_children():
            w.destroy()
        if not query:
            return

        msgs = self.parent.messages.get(self.parent.current_channel, [])
        results = []
        for m in msgs:
            text = (m.get("content", "") or "").lower()
            hit = query in text
            # Chercher aussi dans les pièces jointes "analysis" (ticker/summary)
            if not hit:
                for att in m.get("attachments", []) or []:
                    if att.get("type") == "analysis":
                        if query in (att.get("ticker", "").lower() + " " + att.get("summary", "").lower()):
                            hit = True
                            break
            if hit:
                results.append(m)

        if not results:
            ctk.CTkLabel(self.results_frame, text="Aucun résultat.").pack(pady=8)
            return

        for m in results[-100:]:
            f = ctk.CTkFrame(self.results_frame)
            f.pack(fill="x", padx=6, pady=6)
            ts = m.get("timestamp", "")
            try:
                ts_txt = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts_txt = ts
            preview = (m.get("content") or "[Pièce jointe]")
            ctk.CTkLabel(f, text=f"{m.get('author','?')} — {ts_txt}", font=("Arial", 12, "bold")).pack(anchor="w", padx=8, pady=(8, 0))
            ctk.CTkLabel(f, text=preview[:200], wraplength=460, justify="left").pack(anchor="w", padx=8, pady=(2, 10))


# === CLASSES AVANCÉES POUR LES AMÉLIORATIONS ===

class ChatDatabase:
    """Gestionnaire de base de données SQLite pour le chat"""
    
    def __init__(self, db_path: str = "data/community/chat.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialise les tables de la base de données"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    avatar_path TEXT,
                    status TEXT DEFAULT 'online',
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS channels (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    type TEXT DEFAULT 'public',
                    created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (created_by) REFERENCES users(username)
                );
                
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    author TEXT NOT NULL,
                    content TEXT,
                    message_type TEXT DEFAULT 'text',
                    reply_to TEXT,
                    edited_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (channel_id) REFERENCES channels(id),
                    FOREIGN KEY (author) REFERENCES users(username),
                    FOREIGN KEY (reply_to) REFERENCES messages(id)
                );
                
                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    file_type TEXT,
                    file_size INTEGER,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id);
                CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
            """)
    
    def add_user(self, username: str, email: str = None) -> bool:
        """Ajoute un utilisateur"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO users (username, email) VALUES (?, ?)",
                    (username, email)
                )
                return conn.total_changes > 0
        except Exception as e:
            print(f"Erreur ajout utilisateur: {e}")
            return False
    
    def get_messages(self, channel_id: str, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Récupère les messages d'un canal"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT m.*, 
                           GROUP_CONCAT(a.filename) as attachments
                    FROM messages m
                    LEFT JOIN attachments a ON m.id = a.message_id
                    WHERE m.channel_id = ?
                    GROUP BY m.id
                    ORDER BY m.created_at DESC
                    LIMIT ? OFFSET ?
                """, (channel_id, limit, offset))
                
                messages = []
                for row in cursor.fetchall():
                    msg = dict(row)
                    if msg['attachments']:
                        msg['attachments'] = msg['attachments'].split(',')
                    else:
                        msg['attachments'] = []
                    messages.append(msg)
                
                return list(reversed(messages))
        except Exception as e:
            print(f"Erreur récupération messages: {e}")
            return []
    
    def add_message(self, message_data: Dict) -> bool:
        """Ajoute un message"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO messages (id, channel_id, author, content, message_type, reply_to)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    message_data['id'],
                    message_data['channel_id'],
                    message_data['author'],
                    message_data.get('content', ''),
                    message_data.get('type', 'text'),
                    message_data.get('reply_to')
                ))
                
                # Ajouter les attachements
                for attachment in message_data.get('attachments', []):
                    conn.execute("""
                        INSERT INTO attachments (id, message_id, filename, original_name, file_type, file_size)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        str(uuid.uuid4()),
                        message_data['id'],
                        attachment.get('filename', ''),
                        attachment.get('original_name', ''),
                        attachment.get('type', ''),
                        attachment.get('size', 0)
                    ))
                
                return True
        except Exception as e:
            print(f"Erreur ajout message: {e}")
            return False


class NotificationManager:
    """Gestionnaire de notifications pour le chat"""
    
    def __init__(self, chat_instance):
        self.chat = chat_instance
        self.notification_queue = []
        self.sound_enabled = True
        
    def add_notification(self, notification_type: str, message: str, user: str = None):
        """Ajoute une notification"""
        notification = {
            'type': notification_type,
            'message': message,
            'user': user,
            'timestamp': datetime.now(),
            'read': False
        }
        self.notification_queue.append(notification)
        
        # Afficher la notification
        self._show_notification(notification)
    
    def _show_notification(self, notification):
        """Affiche une notification système"""
        try:
            # Tentative d'import de la fonction de notification
            try:
                from main_app import safe_show_notification
                if notification['type'] == 'mention':
                    safe_show_notification(f"@{notification['user']}: {notification['message'][:50]}...", "info")
                elif notification['type'] == 'message':
                    safe_show_notification(f"Nouveau message de {notification['user']}", "info")
                elif notification['type'] == 'join':
                    safe_show_notification(f"{notification['user']} a rejoint le chat", "success")
            except ImportError:
                print(f"Notification: {notification['type']} - {notification['message']}")
            
            # Son de notification
            if self.sound_enabled:
                self._play_notification_sound()
                
        except Exception as e:
            print(f"Erreur notification: {e}")
    
    def _play_notification_sound(self):
        """Joue un son de notification"""
        try:
            import pygame
            pygame.mixer.init()
            # Beep simple
            pygame.mixer.Sound.play(pygame.mixer.Sound(b'\x00\x00'))
        except:
            try:
                import winsound
                winsound.Beep(800, 200)
            except:
                pass


class ModerationSystem:
    """Système de modération pour le chat"""
    
    def __init__(self):
        self.banned_words = ['spam', 'scam', 'arnaque']
        self.user_warnings = {}
        self.rate_limits = {}
        
    def moderate_message(self, message: str, user: str) -> Dict:
        """Modère un message"""
        result = {
            'allowed': True,
            'reason': None,
            'filtered_message': message
        }
        
        # Vérifier le rate limiting
        if self._check_rate_limit(user):
            result['allowed'] = False
            result['reason'] = "Trop de messages envoyés rapidement"
            return result
        
        # Filtrer les mots interdits
        filtered_message = self._filter_banned_words(message)
        if filtered_message != message:
            result['filtered_message'] = filtered_message
            self._add_warning(user, "Langage inapproprié")
        
        return result
    
    def _check_rate_limit(self, user: str, max_messages: int = 5, window: int = 60) -> bool:
        """Vérifie le rate limiting"""
        now = time.time()
        
        if user not in self.rate_limits:
            self.rate_limits[user] = []
        
        # Nettoyer les anciens timestamps
        self.rate_limits[user] = [
            ts for ts in self.rate_limits[user] 
            if now - ts < window
        ]
        
        # Vérifier la limite
        if len(self.rate_limits[user]) >= max_messages:
            return True
        
        # Ajouter le timestamp actuel
        self.rate_limits[user].append(now)
        return False
    
    def _filter_banned_words(self, message: str) -> str:
        """Filtre les mots interdits"""
        filtered = message
        for word in self.banned_words:
            filtered = filtered.replace(word, '*' * len(word))
        return filtered
    
    def _add_warning(self, user: str, reason: str):
        """Ajoute un avertissement à un utilisateur"""
        if user not in self.user_warnings:
            self.user_warnings[user] = []
        
        self.user_warnings[user].append({
            'reason': reason,
            'timestamp': datetime.now()
        })


# === CLASSE ENHANCED APRÈS LA CLASSE DE BASE ===
class EnhancedCommunityChat(CommunityChat):
    """Version améliorée du chat communautaire"""
    
    def __init__(self, parent, user_profile):
        # Initialiser les nouveaux composants
        self.db = ChatDatabase()
        self.notifications = NotificationManager(self)
        self.moderation = ModerationSystem()
        self.typing_indicators = {}
        self.online_users = set()
        self.users_dynamic_frame = None  # frame dynamique pour la liste des utilisateurs
        
        super().__init__(parent, user_profile)
        
        # Ajouter l'utilisateur à la base
        self.db.add_user(self.user_profile.get('username', 'Utilisateur'))
        
        # Injecter/placer une section utilisateurs dynamique (en plus de la statique de base)
        self._ensure_dynamic_users_section()
        
        # Démarrer les services
        self.start_services()
    
    # ---------- Surcouches de comportement ----------
    def display_messages(self):
        """Override : affiche depuis la base (et pas depuis les fichiers JSON)."""
        self.refresh_messages()
    
    def add_attachment_to_message(self, attachment):
        """Override : ajoute une pièce jointe en base également."""
        try:
            username = self.user_profile.get("username", "Utilisateur")
            msg_id = self.generate_message_id()
            message_data = {
                "id": msg_id,
                "channel_id": self.current_channel,
                "author": username,
                "content": "",
                "type": "attachment",
                "attachments": [attachment],
                "reply_to": None
            }
            if self.db.add_message(message_data):
                self.refresh_messages()
        except Exception as e:
            print(f"Erreur ajout pièce jointe (DB): {e}")
            # fallback en fichiers JSON
            super().add_attachment_to_message(attachment)
    
    def send_message(self, event=None):
        """Version améliorée d'envoi de message avec modération (DB)."""
        content = self.message_entry.get().strip()
        if not content:
            return
        
        username = self.user_profile.get("username", "Utilisateur")
        
        # Modération du message
        moderation_result = self.moderation.moderate_message(content, username)
        
        if not moderation_result['allowed']:
            try:
                from main_app import safe_show_notification
                safe_show_notification(f"Message bloqué: {moderation_result['reason']}", "warning")
            except ImportError:
                messagebox.showwarning("Message bloqué", moderation_result['reason'])
            return
        
        # Utiliser le message filtré
        filtered_content = moderation_result['filtered_message']
        
        # Créer le message
        message_data = {
            "id": self.generate_message_id(),
            "channel_id": self.current_channel,
            "author": username,
            "content": filtered_content,
            "type": "text",
            "attachments": [],
            "reply_to": None
        }
        
        # Sauvegarder en base
        if self.db.add_message(message_data):
            # Effacer le champ de saisie
            self.message_entry.delete(0, "end")
            
            # Actualiser l'affichage
            self.refresh_messages()
            
            # Notification pour les autres utilisateurs
            self.notifications.add_notification(
                'message', 
                filtered_content[:50], 
                username
            )
    
    # ---------- Services & helpers ----------
    def start_services(self):
        """Démarre les services en arrière-plan"""
        # Service de mise à jour des utilisateurs en ligne
        threading.Thread(target=self._update_online_users, daemon=True).start()
        
        # Service de nettoyage des indicateurs de frappe
        threading.Thread(target=self._cleanup_typing_indicators, daemon=True).start()
    
    def refresh_messages(self):
        """Actualise les messages depuis la base de données"""
        try:
            messages = self.db.get_messages(self.current_channel, limit=50)
            
            # Nettoyer l'affichage
            for widget in self.messages_area.winfo_children():
                widget.destroy()
            
            # Afficher les messages
            for msg in messages:
                self.create_enhanced_message_widget(msg)
            
            # Scroll vers le bas
            try:
                self.messages_area._parent_canvas.yview_moveto(1.0)
            except:
                pass
                
        except Exception as e:
            print(f"Erreur actualisation messages: {e}")
    
    def create_enhanced_message_widget(self, message):
        """Crée un widget de message amélioré"""
        # Container du message
        msg_container = ctk.CTkFrame(self.messages_area, fg_color="transparent")
        msg_container.pack(fill="x", pady=5, padx=10)
        
        # Header avec avatar et infos
        header_frame = ctk.CTkFrame(msg_container, fg_color="transparent")
        header_frame.pack(fill="x")
        
        # Avatar (simulé avec emoji)
        avatar = self._get_user_avatar(message.get("author", "?"))
        ctk.CTkLabel(
            header_frame,
            text=avatar,
            font=("Arial", 16)
        ).pack(side="left", padx=(0, 10))
        
        # Nom et timestamp
        info_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True)
        
        user_color = self.get_user_color(message.get("author", "?"))
        ctk.CTkLabel(
            info_frame,
            text=message.get('author', '?'),
            font=("Arial", 12, "bold"),
            text_color=user_color
        ).pack(side="left")
        
        # Timestamp formaté
        try:
            dt = datetime.fromisoformat(message.get('created_at', ''))
            timestamp = dt.strftime("%H:%M")
        except:
            timestamp = "--:--"
        
        ctk.CTkLabel(
            info_frame,
            text=timestamp,
            font=("Arial", 10),
            text_color="#888888"
        ).pack(side="left", padx=(10, 0))
        
        # Actions du message
        actions_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        actions_frame.pack(side="right")
        
        # Bouton réaction
        react_btn = ctk.CTkButton(
            actions_frame,
            text="😊",
            width=25,
            height=20,
            command=lambda: self.add_reaction(message.get('id'), "😊")
        )
        react_btn.pack(side="left", padx=2)
        
        # Bouton répondre
        reply_btn = ctk.CTkButton(
            actions_frame,
            text="↩",
            width=25,
            height=20,
            command=lambda: self.reply_to_message(message.get('id'))
        )
        reply_btn.pack(side="left", padx=2)
        
        # Contenu du message
        content_frame = ctk.CTkFrame(msg_container, fg_color="#2a2d36")
        content_frame.pack(fill="x", pady=(5, 0))
        
        # Texte du message
        if message.get('content'):
            content_label = ctk.CTkLabel(
                content_frame,
                text=message['content'],
                font=("Arial", 12),
                text_color="#FFFFFF",
                wraplength=500,
                justify="left"
            )
            content_label.pack(anchor="w", padx=15, pady=10)
    
    def add_reaction(self, message_id: str, emoji: str):
        """Ajoute une réaction à un message"""
        try:
            print(f"Réaction {emoji} ajoutée au message {message_id}")
            try:
                from main_app import safe_show_notification
                safe_show_notification("Réaction ajoutée!", "success")
            except ImportError:
                print("Réaction ajoutée!")
        except Exception as e:
            print(f"Erreur ajout réaction: {e}")
    
    def reply_to_message(self, message_id: str):
        """Prépare une réponse à un message"""
        try:
            print(f"Réponse au message {message_id}")
            self.message_entry.focus_set()
        except Exception as e:
            print(f"Erreur préparation réponse: {e}")
    
    def _get_user_avatar(self, username: str) -> str:
        """Retourne un avatar emoji pour l'utilisateur"""
        avatars = ["👤", "🧑", "👨", "👩", "🧔", "👱", "🧓", "👴", "👵"]
        hash_val = hash(username) % len(avatars)
        return avatars[hash_val]
    
    def _update_online_users(self):
        """Met à jour la liste des utilisateurs en ligne (simulation)"""
        while True:
            try:
                # Simuler des utilisateurs en ligne
                self.online_users = {"TradingPro", "AlphaInvestor", "Vous"}
                
                # Mettre à jour l'affichage
                self.after_idle(self.update_users_display)
                
                time.sleep(30)  # Mise à jour toutes les 30 secondes
            except Exception as e:
                print(f"Erreur mise à jour utilisateurs: {e}")
                time.sleep(60)
    
    def _cleanup_typing_indicators(self):
        """Nettoie périodiquement les indicateurs de frappe périmés"""
        while True:
            try:
                now = time.time()
                expired = [u for u, ts in self.typing_indicators.items() if now - ts > 5]
                for u in expired:
                    self.typing_indicators.pop(u, None)
                time.sleep(3)
            except Exception:
                time.sleep(5)
    
    def _ensure_dynamic_users_section(self):
        """Crée une section dynamique pour la liste des utilisateurs, si absente."""
        if self.users_dynamic_frame and self.users_dynamic_frame.winfo_exists():
            return
        # Ajoute un bloc supplémentaire en bas de la sidebar pour la version dynamique
        self.users_dynamic_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.users_dynamic_frame.pack(fill="x", padx=15, pady=(10, 15))
    
    def update_users_display(self):
        """Met à jour l'affichage des utilisateurs en ligne (section dynamique)."""
        try:
            self._ensure_dynamic_users_section()
            f = self.users_dynamic_frame
            
            # Nettoyer
            for w in f.winfo_children():
                try:
                    w.destroy()
                except:
                    pass
            
            # Titre / séparateur
            sep = ctk.CTkFrame(f, height=1, fg_color="#333333")
            sep.pack(fill="x", pady=(0, 8))
            
            count = len(self.online_users)
            title = ctk.CTkLabel(
                f, text=f"👥 En ligne ({count})",
                font=("Arial", 12, "bold"),
                text_color="#888888"
            )
            title.pack(anchor="w", pady=(0, 6))
            
            # Liste
            for user in sorted(self.online_users):
                row = ctk.CTkFrame(f, fg_color="#2a2d36")
                row.pack(fill="x", pady=2)
                color = "#00FF88" if user != "Vous" else "#00BFFF"
                ctk.CTkLabel(row, text=f"🟢 {user}", font=("Arial", 10), text_color=color)\
                    .pack(anchor="w", padx=10, pady=5)
            
            # Indicateurs de frappe (optionnel)
            if self.typing_indicators:
                typing_line = ", ".join(sorted(self.typing_indicators.keys()))
                ctk.CTkLabel(
                    f,
                    text=f"✍ {typing_line} est en train d'écrire...",
                    font=("Arial", 10),
                    text_color="#aaaaaa"
                ).pack(anchor="w", pady=(6, 0))
        
        except Exception as e:
            print(f"Erreur update_users_display: {e}")
