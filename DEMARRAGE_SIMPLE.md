# 🚀 Démarrage Simple de HelixOne avec l'Analyse Complète

**Tout est prêt !** Suivez ces 3 étapes simples :

---

## ÉTAPE 1️⃣ : Démarrer le Backend

### Option A (Recommandé) : Script automatique

```bash
cd /Users/macintosh/Desktop/helixone
./START_BACKEND.sh
```

### Option B : Manuel

```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend
../venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### ✅ Attendez de voir :

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**⚠️ NE FERMEZ PAS ce terminal !** Laissez-le tourner.

---

## ÉTAPE 2️⃣ : Démarrer l'Interface

**Ouvrez UN NOUVEAU terminal** et lancez :

### Option A (Recommandé) : Script automatique

```bash
cd /Users/macintosh/Desktop/helixone
./START_INTERFACE.sh
```

### Option B : Manuel

```bash
cd /Users/macintosh/Desktop/helixone
./venv/bin/python -m src.main
```

---

## ÉTAPE 3️⃣ : Tester l'Analyse Complète

Dans l'interface HelixOne qui s'ouvre :

1. **Connectez-vous** avec votre compte

2. **Allez dans "🔍 Recherche"** (menu de gauche)

3. **Tapez "AAPL"** dans la barre de recherche

4. **Cliquez sur "Analyser"**

5. **Attendez 5-10 secondes**

6. **Regardez l'onglet "🔍 Analyse"**

---

## ✨ Ce que vous DEVEZ voir

Si tout fonctionne, vous verrez :

### 🟢 Badge vert en haut
```
✨ ANALYSE COMPLÈTE 8 ÉTAPES
```

### 📋 Sections affichées :
- **Executive Summary** (nouveau !)
- **Health Score + Recommandation**
- **🚨 Alertes** (Critical/Important/Info) (nouveau !)
- **🧠 Prédictions ML** (1j/3j/7j détaillées)
- **💭 Analyse Sentiment** (trend, velocity) (nouveau !)
- **📅 Événements à venir** (Fed, earnings, etc.) (nouveau !)
- **📡 Sources de données** (35+ sources) (nouveau !)
- **📊 Métriques de position**

---

## ❌ Si ça ne marche pas

### Backend ne démarre pas ?

**1. Vérifier les erreurs dans le terminal backend**

Si vous voyez `ERROR:`, lisez le message et :
- Vérifiez que le port 8000 est libre : `lsof -ti:8000`
- Si occupé : `lsof -ti:8000 | xargs kill -9`

**2. Relancer avec cache nettoyé**

```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend
find . -name "*.pyc" -delete
find . -type d -name __pycache__ -exec rm -rf {} +
../venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Interface ne démarre pas ?

**Erreur "No module named 'src'" ?**

Vérifiez que vous lancez bien :
```bash
cd /Users/macintosh/Desktop/helixone
./venv/bin/python -m src.main
```

**PAS** :
```bash
# ❌ INCORRECT
./venv/bin/python src/main.py
```

### Analyse complète ne s'affiche pas ?

**1. Vérifier que le backend répond**

```bash
curl http://127.0.0.1:8000/health
```

Devrait retourner : `{"status":"ok",...}`

**2. Vérifier que l'endpoint existe**

Ouvrez dans votre navigateur :
```
http://localhost:8000/docs
```

Cherchez `/api/analysis/stock-deep-analysis` dans la liste.

**3. Vérifier les logs**

Dans le terminal du **backend**, cherchez :
```
INFO:     127.0.0.1 - "POST /api/analysis/stock-deep-analysis HTTP/1.1"
```

Si vous voyez ça, l'endpoint est appelé.

**4. Fallback automatique**

Si l'analyse complète échoue, l'interface utilise automatiquement l'analyse standard (l'ancienne).
Vous NE verrez PAS le badge "8 ÉTAPES" mais l'analyse basique fonctionnera quand même.

---

## 🔍 Vérification Rapide

### Est-ce que l'analyse complète fonctionne ?

**OUI si vous voyez** :
- ✅ Badge vert "✨ ANALYSE COMPLÈTE 8 ÉTAPES"
- ✅ Section "📋 Executive Summary"
- ✅ Section "🚨 Alertes"
- ✅ Section "💭 Analyse Sentiment"
- ✅ Section "📅 Événements à venir"

**NON (fallback standard) si vous voyez** :
- ❌ Pas de badge "8 ÉTAPES"
- ❌ Seulement Health Score et Recommandation
- ❌ Pas de sections supplémentaires

Dans ce cas, regardez les logs du backend pour voir l'erreur.

---

## 📞 Aide Supplémentaire

**Fichiers utiles** :
- `STATUS_INTEGRATION_ANALYSE.md` - Status complet de l'intégration
- `ANALYSE_COMPLETE_RECHERCHE.md` - Documentation technique détaillée

**Logs** :
- Backend : Dans le terminal où vous avez lancé `START_BACKEND.sh`
- Frontend : Dans le terminal où vous avez lancé `START_INTERFACE.sh`

---

## 🎯 Résumé Ultra-Rapide

```bash
# Terminal 1 : Backend
cd /Users/macintosh/Desktop/helixone
./START_BACKEND.sh

# Terminal 2 : Interface (ouvrir un NOUVEAU terminal)
cd /Users/macintosh/Desktop/helixone
./START_INTERFACE.sh

# Dans l'interface
1. Connexion
2. Recherche → Taper "AAPL" → Analyser
3. Onglet "Analyse" → Voir le badge "8 ÉTAPES" ✨
```

---

**C'est tout !** Vous avez maintenant accès à l'analyse complète 8 étapes directement dans l'onglet Recherche ! 🎉
