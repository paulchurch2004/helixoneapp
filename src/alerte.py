import customtkinter as ctk
import threading
import time
from src.donnees_actions import actions_db
from tkinter import scrolledtext
from datetime import datetime

# Options de taille de scan
scan_size_options = {
    "50 actions (~1 min)": 50,
    "100 actions (~2 min)": 100,
    "200 actions (~4 min)": 200,
    "400 actions (~8 min)": 400,
    "Intégrale (~10+ min)": len(actions_db)
}

def create_zone_alerte_frame(parent):
    frame = ctk.CTkFrame(parent)
    frame.pack(fill="both", expand=True, padx=20, pady=20)

    top_frame = ctk.CTkFrame(frame)
    top_frame.pack(pady=10)

    scan_mode = ctk.StringVar(value="50 actions (~1 min)")

    ctk.CTkLabel(top_frame, text="Sélection du nombre d’actions à scanner :").grid(row=0, column=0, padx=10)
    ctk.CTkOptionMenu(top_frame, variable=scan_mode, values=list(scan_size_options.keys())).grid(row=0, column=1, padx=10)

    status_label = ctk.CTkLabel(frame, text="🟢 Prêt à lancer l’analyse")
    status_label.pack(pady=5)

    progress_bar = ctk.CTkProgressBar(frame, orientation="horizontal", width=500)
    progress_bar.set(0)
    progress_bar.pack(pady=5)

    results_box = scrolledtext.ScrolledText(frame, wrap="word", height=30, font=("Consolas", 11), bg="#1a1a1a", fg="white")
    results_box.pack(pady=10, fill="both", expand=True)

    scan_thread = None
    is_scanning = False
    stop_signal = False

    def toggle_scan():
        nonlocal scan_thread, is_scanning, stop_signal

        if not is_scanning:
            stop_signal = False
            is_scanning = True
            start_btn.configure(text="⏹️ Arrêter l’analyse")
            status_label.configure(text="🔍 Analyse en cours...")

            scan_thread = threading.Thread(
                target=run_scan,
                args=(scan_size_options[scan_mode.get()], progress_bar, results_box, status_label),
                daemon=True
            )
            scan_thread.start()
        else:
            stop_signal = True
            is_scanning = False
            start_btn.configure(text="▶️ Lancer l’analyse")
            status_label.configure(text="⏸️ Analyse interrompue par l’utilisateur")

    start_btn = ctk.CTkButton(frame, text="▶️ Lancer l’analyse", command=toggle_scan)
    start_btn.pack(pady=10)

def run_scan(limit, progress_bar, results_box, status_label):
    # ✅ Import évité en haut du fichier pour contourner l'import circulaire
    from src.engine.analyseur import get_stock_data

    tickers = list(actions_db.items())[:limit]
    results_box.delete("1.0", "end")
    results = []
    total = len(tickers)

    for i, (name, ticker) in enumerate(tickers, 1):
        try:
            stock, hist, info, fondamentaux, macro, score, verdict, signaux, *_ = get_stock_data(ticker, mode="Court Terme")
            prix = fondamentaux.get("prix")
            atr = hist["ATR"].iloc[-1] if "ATR" in hist else None

            if score >= 70 or any("⚡" in s or "✅" in s for s in signaux):
                if prix and atr:
                    tp = prix + 1.5 * atr
                    sl = prix - 1.0 * atr
                    net = (tp - prix) / prix

                    ligne = (
                        f"🚨 Signal détecté sur {ticker} ({info.get('shortName', name)})\n"
                        f"• Score FXI : {score}/100\n"
                        f"• Prix actuel : {prix:.2f} €\n"
                        f"• Signal(s) : {', '.join(signaux)}\n"
                        f"• TP : {tp:.2f} € | SL : {sl:.2f} €\n"
                        f"• Rendement net estimé : {net*100:.2f} %\n"
                        f"{'-'*60}\n"
                    )
                    results.append(ligne)
        except Exception as e:
            print(f"[Erreur] {ticker} : {e}")
            continue

        progress_bar.set(i / total)
        time.sleep(0.05)

    if not results:
        results_box.insert("end", "✅ Aucun signal détecté dans cette session.\n")
    else:
        results_box.insert("end", "\n".join(results))

    progress_bar.set(1)
    status_label.configure(text=f"✅ Analyse terminée à {datetime.now().strftime('%H:%M:%S')}")
