# 📱 Frontend - Composant Onglet Alertes

Ce document contient le code frontend pour afficher les alertes de portefeuille dans votre application.

## 🎯 Composants React

### 1. Service API (alertsService.js)

```javascript
// services/alertsService.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class AlertsService {
  /**
   * Récupère les alertes
   * @param {Object} filters - Filtres optionnels
   * @returns {Promise<Array>} Liste des alertes
   */
  async getAlerts(filters = {}) {
    const params = new URLSearchParams();

    if (filters.severity) params.append('severity', filters.severity);
    if (filters.status) params.append('status', filters.status);
    if (filters.ticker) params.append('ticker', filters.ticker);
    if (filters.days) params.append('days', filters.days);
    if (filters.limit) params.append('limit', filters.limit);

    const response = await axios.get(`${API_BASE_URL}/api/portfolio/alerts?${params}`);
    return response.data;
  }

  /**
   * Récupère une alerte spécifique
   */
  async getAlert(alertId) {
    const response = await axios.get(`${API_BASE_URL}/api/portfolio/alerts/${alertId}`);
    return response.data;
  }

  /**
   * Met à jour le statut d'une alerte
   */
  async updateAlertStatus(alertId, status) {
    const response = await axios.patch(
      `${API_BASE_URL}/api/portfolio/alerts/${alertId}`,
      { status }
    );
    return response.data;
  }

  /**
   * Récupère le nombre d'alertes non lues
   */
  async getUnreadCount() {
    const response = await axios.get(`${API_BASE_URL}/api/portfolio/alerts/unread/count`);
    return response.data;
  }

  /**
   * Récupère les recommandations
   */
  async getRecommendations(filters = {}) {
    const params = new URLSearchParams();

    if (filters.action) params.append('action', filters.action);
    if (filters.ticker) params.append('ticker', filters.ticker);
    if (filters.min_confidence) params.append('min_confidence', filters.min_confidence);
    if (filters.days) params.append('days', filters.days);
    if (filters.limit) params.append('limit', filters.limit);

    const response = await axios.get(`${API_BASE_URL}/api/portfolio/recommendations?${params}`);
    return response.data;
  }

  /**
   * Récupère le résumé du dashboard
   */
  async getDashboardSummary() {
    const response = await axios.get(`${API_BASE_URL}/api/portfolio/dashboard/summary`);
    return response.data;
  }

  /**
   * Récupère l'historique des analyses
   */
  async getAnalysisHistory(filters = {}) {
    const params = new URLSearchParams();

    if (filters.analysis_time) params.append('analysis_time', filters.analysis_time);
    if (filters.days) params.append('days', filters.days);
    if (filters.limit) params.append('limit', filters.limit);

    const response = await axios.get(`${API_BASE_URL}/api/portfolio/history?${params}`);
    return response.data;
  }
}

export default new AlertsService();
```

---

### 2. Composant Principal - Onglet Alertes (AlertsTab.jsx)

```jsx
// components/AlertsTab.jsx
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  IconButton,
  Button,
  Tabs,
  Tab,
  Badge,
  CircularProgress,
  Alert,
  Divider,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CheckCircle as CheckIcon,
  Close as CloseIcon,
  TrendingUp as TrendingUpIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Notifications as NotificationsIcon
} from '@mui/icons-material';
import alertsService from '../services/alertsService';
import ReactMarkdown from 'react-markdown';

// Couleurs par sévérité
const SEVERITY_COLORS = {
  critical: '#d32f2f',    // Rouge
  warning: '#f57c00',     // Orange
  opportunity: '#388e3c', // Vert
  info: '#1976d2'         // Bleu
};

// Icônes par sévérité
const SEVERITY_ICONS = {
  critical: <WarningIcon />,
  warning: <WarningIcon />,
  opportunity: <TrendingUpIcon />,
  info: <InfoIcon />
};

// Composant pour une carte d'alerte
const AlertCard = ({ alert, onUpdateStatus }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleMarkAsRead = async () => {
    await onUpdateStatus(alert.id, 'read');
  };

  const handleMarkAsActed = async () => {
    await onUpdateStatus(alert.id, 'acted');
  };

  const handleDismiss = async () => {
    await onUpdateStatus(alert.id, 'dismissed');
  };

  return (
    <Card
      sx={{
        mb: 2,
        borderLeft: `4px solid ${SEVERITY_COLORS[alert.severity]}`,
        backgroundColor: alert.status === 'new' ? '#fff8e1' : 'white'
      }}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="start">
          <Box flex={1}>
            {/* En-tête */}
            <Box display="flex" alignItems="center" gap={1} mb={1}>
              <Box sx={{ color: SEVERITY_COLORS[alert.severity] }}>
                {SEVERITY_ICONS[alert.severity]}
              </Box>
              <Typography variant="h6" component="div">
                {alert.title}
              </Typography>
              <Chip
                label={alert.severity.toUpperCase()}
                size="small"
                sx={{
                  backgroundColor: SEVERITY_COLORS[alert.severity],
                  color: 'white',
                  fontWeight: 'bold'
                }}
              />
              {alert.ticker && (
                <Chip label={alert.ticker} size="small" variant="outlined" />
              )}
              {alert.status === 'new' && (
                <Chip label="NOUVEAU" size="small" color="primary" />
              )}
            </Box>

            {/* Message */}
            <Box
              sx={{
                maxHeight: isExpanded ? 'none' : '100px',
                overflow: 'hidden',
                cursor: 'pointer'
              }}
              onClick={() => setIsExpanded(!isExpanded)}
            >
              <ReactMarkdown>{alert.message}</ReactMarkdown>
            </Box>

            {/* Action requise */}
            {alert.action_required && (
              <Box mt={2} p={2} sx={{ backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                <Typography variant="subtitle2" fontWeight="bold">
                  📌 Action recommandée:
                </Typography>
                <Typography variant="body2">{alert.action_required}</Typography>
              </Box>
            )}

            {/* Confiance */}
            {alert.confidence && (
              <Box mt={1}>
                <Typography variant="caption" color="text.secondary">
                  Confiance: {alert.confidence}%
                </Typography>
              </Box>
            )}

            {/* Timestamp */}
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              {new Date(alert.created_at).toLocaleString('fr-FR')}
            </Typography>
          </Box>

          {/* Actions */}
          <Stack direction="row" spacing={1}>
            {alert.status === 'new' && (
              <IconButton
                size="small"
                onClick={handleMarkAsRead}
                title="Marquer comme lu"
                color="primary"
              >
                <CheckIcon />
              </IconButton>
            )}
            {(alert.status === 'new' || alert.status === 'read') && (
              <Button
                size="small"
                variant="contained"
                color="success"
                onClick={handleMarkAsActed}
              >
                ✓ Traité
              </Button>
            )}
            <IconButton
              size="small"
              onClick={handleDismiss}
              title="Ignorer"
              color="error"
            >
              <CloseIcon />
            </IconButton>
          </Stack>
        </Box>
      </CardContent>
    </Card>
  );
};

// Composant principal
export default function AlertsTab() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [unreadCount, setUnreadCount] = useState(0);
  const [tabValue, setTabValue] = useState(0);
  const [severityFilter, setSeverityFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');

  // Charger les alertes
  const loadAlerts = async () => {
    setLoading(true);
    try {
      const filters = {};
      if (severityFilter) filters.severity = severityFilter;
      if (statusFilter) filters.status = statusFilter;
      filters.days = 7;
      filters.limit = 50;

      const data = await alertsService.getAlerts(filters);
      setAlerts(data);

      // Charger le compteur non lus
      const countData = await alertsService.getUnreadCount();
      setUnreadCount(countData.unread_count);

      setError(null);
    } catch (err) {
      setError('Erreur lors du chargement des alertes');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Mettre à jour le statut d'une alerte
  const handleUpdateStatus = async (alertId, newStatus) => {
    try {
      await alertsService.updateAlertStatus(alertId, newStatus);
      // Recharger les alertes
      await loadAlerts();
    } catch (err) {
      console.error('Erreur mise à jour statut:', err);
    }
  };

  useEffect(() => {
    loadAlerts();
    // Rafraîchir toutes les 60 secondes
    const interval = setInterval(loadAlerts, 60000);
    return () => clearInterval(interval);
  }, [severityFilter, statusFilter]);

  // Filtrer les alertes par onglet
  const getFilteredAlerts = () => {
    switch (tabValue) {
      case 0: // Toutes
        return alerts;
      case 1: // Non lues
        return alerts.filter(a => a.status === 'new');
      case 2: // Critiques
        return alerts.filter(a => a.severity === 'critical');
      case 3: // Opportunités
        return alerts.filter(a => a.severity === 'opportunity');
      default:
        return alerts;
    }
  };

  const filteredAlerts = getFilteredAlerts();

  return (
    <Box sx={{ p: 3 }}>
      {/* En-tête */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" gutterBottom>
            <NotificationsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Alertes de Portefeuille
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Alertes et recommandations générées automatiquement 2x/jour (7h00 + 17h00 EST)
          </Typography>
        </Box>
        <IconButton onClick={loadAlerts} color="primary">
          <RefreshIcon />
        </IconButton>
      </Box>

      {/* Filtres */}
      <Box display="flex" gap={2} mb={3}>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Sévérité</InputLabel>
          <Select
            value={severityFilter}
            label="Sévérité"
            onChange={(e) => setSeverityFilter(e.target.value)}
          >
            <MenuItem value="">Toutes</MenuItem>
            <MenuItem value="critical">Critique</MenuItem>
            <MenuItem value="warning">Avertissement</MenuItem>
            <MenuItem value="opportunity">Opportunité</MenuItem>
            <MenuItem value="info">Info</MenuItem>
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Statut</InputLabel>
          <Select
            value={statusFilter}
            label="Statut"
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <MenuItem value="">Tous</MenuItem>
            <MenuItem value="new">Nouveaux</MenuItem>
            <MenuItem value="read">Lus</MenuItem>
            <MenuItem value="acted">Traités</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Onglets */}
      <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} sx={{ mb: 3 }}>
        <Tab label={`Toutes (${alerts.length})`} />
        <Tab
          label={
            <Badge badgeContent={unreadCount} color="error">
              Non lues
            </Badge>
          }
        />
        <Tab label="Critiques" />
        <Tab label="Opportunités" />
      </Tabs>

      <Divider sx={{ mb: 3 }} />

      {/* Contenu */}
      {loading ? (
        <Box display="flex" justifyContent="center" py={4}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error">{error}</Alert>
      ) : filteredAlerts.length === 0 ? (
        <Alert severity="info">Aucune alerte à afficher</Alert>
      ) : (
        <Box>
          {filteredAlerts.map((alert) => (
            <AlertCard
              key={alert.id}
              alert={alert}
              onUpdateStatus={handleUpdateStatus}
            />
          ))}
        </Box>
      )}
    </Box>
  );
}
```

---

### 3. Composant Badge de notification (AlertsBadge.jsx)

Pour afficher le compteur d'alertes non lues dans la barre de navigation :

```jsx
// components/AlertsBadge.jsx
import React, { useState, useEffect } from 'react';
import { Badge, IconButton } from '@mui/material';
import { Notifications as NotificationsIcon } from '@mui/icons-material';
import alertsService from '../services/alertsService';

export default function AlertsBadge({ onClick }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const loadCount = async () => {
      try {
        const data = await alertsService.getUnreadCount();
        setCount(data.unread_count);
      } catch (err) {
        console.error('Erreur chargement compteur:', err);
      }
    };

    loadCount();
    // Rafraîchir toutes les 30 secondes
    const interval = setInterval(loadCount, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <IconButton color="inherit" onClick={onClick}>
      <Badge badgeContent={count} color="error">
        <NotificationsIcon />
      </Badge>
    </IconButton>
  );
}
```

---

## 📦 Installation des dépendances

```bash
npm install axios @mui/material @mui/icons-material react-markdown
```

---

## 🔧 Configuration

1. **Créer le fichier `.env`** :
```env
REACT_APP_API_URL=http://localhost:8000
```

2. **Configurer Axios avec authentification** (si nécessaire) :

```javascript
// services/axios.js
import axios from 'axios';

// Ajouter le token JWT si l'utilisateur est connecté
axios.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default axios;
```

---

## 🎨 Intégration dans l'application

```jsx
// App.js ou votre router
import AlertsTab from './components/AlertsTab';
import AlertsBadge from './components/AlertsBadge';

function App() {
  return (
    <Router>
      <AppBar>
        {/* Badge dans la navigation */}
        <AlertsBadge onClick={() => navigate('/alerts')} />
      </AppBar>

      <Routes>
        {/* Onglet alertes */}
        <Route path="/alerts" element={<AlertsTab />} />
      </Routes>
    </Router>
  );
}
```

---

## 📱 Fonctionnalités

✅ **Affichage des alertes** avec code couleur par sévérité
✅ **Filtres** par sévérité et statut
✅ **Onglets** (Toutes, Non lues, Critiques, Opportunités)
✅ **Badge** de notification avec compteur temps réel
✅ **Actions** : Marquer comme lu, Traité, Ignorer
✅ **Markdown** supporté dans les messages
✅ **Auto-refresh** toutes les 60 secondes
✅ **Responsive** et moderne avec Material-UI

---

## 🚀 Pour aller plus loin

1. **Notifications push** : Intégrer Firebase Cloud Messaging
2. **WebSocket** : Recevoir les alertes en temps réel
3. **Sons** : Émettre un son lors de nouvelles alertes critiques
4. **Animations** : Animer l'arrivée de nouvelles alertes
5. **Export** : Télécharger les alertes en PDF/Excel
