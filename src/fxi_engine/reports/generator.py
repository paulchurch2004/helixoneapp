"""
Générateur de rapports d'analyse avec macro-économie
"""

from datetime import datetime
from typing import TYPE_CHECKING

from ..core.config import EngineConfig

if TYPE_CHECKING:
    from ..core.engine import AnalysisResult

class ReportGenerator:
    """Générateur de rapports professionnels avec analyse macro"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
    
    def generate(self, analysis: "AnalysisResult", format: str = "detailed") -> str:
        """
        Génère un rapport d'analyse
        
        Args:
            analysis: Résultat de l'analyse
            format: Format du rapport ("detailed", "executive", "summary")
        
        Returns:
            str: Rapport formaté
        """
        if format == "executive":
            return self._generate_executive_summary(analysis)
        elif format == "summary":
            return self._generate_summary(analysis)
        else:
            return self._generate_detailed_report(analysis)
    
    def _generate_detailed_report(self, analysis: "AnalysisResult") -> str:
        """Génère un rapport détaillé avec macro-économie"""
        
        # Déterminer les indicateurs de couleur
        score_indicator = self._get_score_indicator(analysis.final_score)
        risk_level = self._get_risk_level(analysis.risk_score)
        macro_level = self._get_macro_level(analysis.macro_score)
        
        return f"""
# 📊 RAPPORT D'ANALYSE FXI v2.0 - {analysis.ticker.upper()}

**Date d'analyse** : {analysis.timestamp.strftime('%d/%m/%Y à %H:%M')}
**Temps d'exécution** : {analysis.execution_time:.2f} secondes
**Qualité des données** : {analysis.data_quality:.0%}

---

## 🎯 RÉSUMÉ EXÉCUTIF

{score_indicator} **Score Global FXI** : **{analysis.final_score:.1f}/100**
{score_indicator} **Recommandation** : **{analysis.recommendation}**
{score_indicator} **Niveau de confiance** : {analysis.confidence:.0f}%
{score_indicator} **Niveau de risque** : {risk_level}

---

## 📈 ANALYSE DÉTAILLÉE PAR DIMENSION

### 🔧 Analyse Technique : {analysis.technical_score:.1f}/100
{self._interpret_technical_score(analysis.technical_score)}

### 💰 Analyse Fondamentale : {analysis.fundamental_score:.1f}/100
{self._interpret_fundamental_score(analysis.fundamental_score)}

### 🎭 Analyse du Sentiment : {analysis.sentiment_score:.1f}/100
{self._interpret_sentiment_score(analysis.sentiment_score)}

### ⚡ Analyse des Risques : {analysis.risk_score:.1f}/100
{self._interpret_risk_score(analysis.risk_score)}

### 🌍 Analyse Macro-Économique : {analysis.macro_score:.1f}/100
{self._interpret_macro_score(analysis.macro_score)}
**Environnement actuel** : {analysis.details.get('macro_environment', 'N/A')}
**Impact sectoriel** : {analysis.details.get('sector_macro_correlation', 'N/A')}

---

## 📋 MÉTRIQUES CLÉS

- **Prix actuel** : {analysis.details.get('current_price', 'N/A')}
- **P/E Ratio** : {analysis.details.get('pe_ratio', 'N/A')}
- **Capitalisation** : {self._format_market_cap(analysis.details.get('market_cap'))}
- **Secteur** : {analysis.details.get('sector', 'N/A')}
- **Industrie** : {analysis.details.get('industry', 'N/A')}

---

## 🎯 RECOMMANDATIONS

{self._generate_recommendations(analysis)}

---

## 📊 PONDÉRATION DES ANALYSES

- Technique : {self.config.technical_weight * 100:.0f}%
- Fondamental : {self.config.fundamental_weight * 100:.0f}%
- Sentiment : {self.config.sentiment_weight * 100:.0f}%
- Risque : {self.config.risk_weight * 100:.0f}%
- Macro-Économie : {self.config.macro_weight * 100:.0f}%

---

## ⚖️ DISCLAIMER

Cette analyse est générée par un système automatisé et ne constitue pas un conseil en investissement personnalisé. Les investissements comportent des risques de perte en capital. Consultez un conseiller financier qualifié avant toute décision d'investissement.

**Sources** : Yahoo Finance, données publiques, indicateurs macro-économiques
**Moteur** : HelixOne FXI Engine v2.0 (avec analyse macro-économique)
"""
    
    def _generate_executive_summary(self, analysis: "AnalysisResult") -> str:
        """Génère un résumé exécutif"""
        score_indicator = self._get_score_indicator(analysis.final_score)
        
        return f"""
# 📊 RÉSUMÉ EXÉCUTIF - {analysis.ticker.upper()}

{score_indicator} **Score** : {analysis.final_score:.1f}/100 | **{analysis.recommendation}** | Confiance : {analysis.confidence:.0f}%

**Scores détaillés** :
- Technique : {analysis.technical_score:.1f}/100
- Fondamental : {analysis.fundamental_score:.1f}/100  
- Sentiment : {analysis.sentiment_score:.1f}/100
- Risque : {analysis.risk_score:.1f}/100
- Macro-Économie : {analysis.macro_score:.1f}/100

**Informations clés** :
- Secteur : {analysis.details.get('sector', 'N/A')}
- Prix : {analysis.details.get('current_price', 'N/A')}
- PE : {analysis.details.get('pe_ratio', 'N/A')}
- Environnement macro : {analysis.details.get('macro_environment', 'N/A')}

*Analyse générée le {analysis.timestamp.strftime('%d/%m/%Y à %H:%M')}*
"""
    
    def _generate_summary(self, analysis: "AnalysisResult") -> str:
        """Génère un résumé court"""
        score_indicator = self._get_score_indicator(analysis.final_score)
        
        return f"""
{analysis.ticker.upper()}: {score_indicator} {analysis.final_score:.1f}/100 - {analysis.recommendation}
Tech: {analysis.technical_score:.1f} | Fund: {analysis.fundamental_score:.1f} | Sent: {analysis.sentiment_score:.1f} | Risk: {analysis.risk_score:.1f} | Macro: {analysis.macro_score:.1f}
Confiance: {analysis.confidence:.0f}% | {analysis.timestamp.strftime('%d/%m %H:%M')}
"""
    
    def _get_score_indicator(self, score: float) -> str:
        """Retourne l'indicateur visuel du score"""
        if score >= 80:
            return "🟢"
        elif score >= 65:
            return "🔵"
        elif score >= 45:
            return "🟡"
        elif score >= 30:
            return "🟠"
        else:
            return "🔴"
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convertit le score de risque en niveau textuel"""
        if risk_score >= 70:
            return "Faible"
        elif risk_score >= 45:
            return "Modéré"
        else:
            return "Élevé"
    
    def _get_macro_level(self, macro_score: float) -> str:
        """Convertit le score macro en niveau textuel"""
        if macro_score >= 70:
            return "Favorable"
        elif macro_score >= 50:
            return "Neutre"
        else:
            return "Défavorable"
    
    def _interpret_technical_score(self, score: float) -> str:
        """Interprète le score technique"""
        if score >= 75:
            return "**Très positif** - Signaux techniques encourageants avec tendance haussière confirmée."
        elif score >= 60:
            return "**Positif** - Situation technique globalement favorable avec quelques signaux de confirmation."
        elif score >= 40:
            return "**Neutre** - Signaux techniques mitigés nécessitant une surveillance accrue."
        else:
            return "**Négatif** - Signaux techniques suggérant la prudence avec tendance baissière."
    
    def _interpret_fundamental_score(self, score: float) -> str:
        """Interprète le score fondamental"""
        if score >= 75:
            return "**Excellents fondamentaux** - Santé financière remarquable avec ratios attractifs."
        elif score >= 60:
            return "**Bons fondamentaux** - Bases financières solides avec valorisation raisonnable."
        elif score >= 40:
            return "**Fondamentaux corrects** - Situation financière acceptable mais quelques points d'attention."
        else:
            return "**Fondamentaux faibles** - Structure financière présentant des risques significatifs."
    
    def _interpret_sentiment_score(self, score: float) -> str:
        """Interprète le score de sentiment"""
        if score >= 70:
            return "**Sentiment très positif** - Consensus des analystes favorable avec momentum institutionnel."
        elif score >= 50:
            return "**Sentiment positif** - Recommandations globalement favorables des professionnels."
        elif score >= 30:
            return "**Sentiment mitigé** - Opinions partagées nécessitant une analyse approfondie."
        else:
            return "**Sentiment négatif** - Consensus défavorable avec prudence recommandée."
    
    def _interpret_risk_score(self, score: float) -> str:
        """Interprète le score de risque"""
        if score >= 70:
            return "**Risque faible** - Profil de risque maîtrisé avec volatilité modérée."
        elif score >= 45:
            return "**Risque modéré** - Quelques facteurs de risque à surveiller."
        else:
            return "**Risque élevé** - Profil de risque préoccupant nécessitant une attention particulière."
    
    def _interpret_macro_score(self, score: float) -> str:
        """Interprète le score macro-économique"""
        if score >= 70:
            return "**Environnement favorable** - Conditions macro-économiques propices avec vents porteurs pour le secteur."
        elif score >= 55:
            return "**Environnement neutre** - Conditions macro-économiques équilibrées sans impact majeur."
        elif score >= 40:
            return "**Environnement mitigé** - Conditions macro-économiques présentant quelques défis sectoriels."
        else:
            return "**Environnement défavorable** - Conditions macro-économiques difficiles pesant sur les perspectives."
    
    def _generate_recommendations(self, analysis: "AnalysisResult") -> str:
        """Génère des recommandations personnalisées incluant la macro"""
        recommendations = []
        
        if analysis.final_score >= 70:
            recommendations.append("• **Position recommandée** : Envisager une prise de position progressive")
            recommendations.append("• **Horizon temporel** : Adapté pour un investissement à moyen terme")
        elif analysis.final_score >= 45:
            recommendations.append("• **Position recommandée** : Surveiller attentivement les prochains développements")
            recommendations.append("• **Horizon temporel** : Attendre des signaux plus clairs")
        else:
            recommendations.append("• **Position recommandée** : Éviter ou réduire l'exposition")
            recommendations.append("• **Horizon temporel** : Situation défavorable à court terme")
        
        # Recommandations basées sur les risques
        if analysis.risk_score < 50:
            recommendations.append("• **Gestion des risques** : Utiliser des stops-loss stricts")
        
        # Recommandations basées sur le technique
        if analysis.technical_score > 70:
            recommendations.append("• **Timing d'entrée** : Momentum technique favorable")
        elif analysis.technical_score < 40:
            recommendations.append("• **Timing d'entrée** : Attendre une amélioration technique")
        
        # Recommandations basées sur la macro
        if analysis.macro_score > 65:
            recommendations.append("• **Contexte macro** : Environnement économique porteur pour ce secteur")
        elif analysis.macro_score < 45:
            recommendations.append("• **Contexte macro** : Attendre une amélioration des conditions économiques")
            recommendations.append("• **Vigilance** : Suivre les annonces macro-économiques à venir")
        
        # Recommandations spécifiques selon secteur et macro
        sector = analysis.details.get('sector', '')
        if sector in ['Technology', 'Real Estate'] and analysis.macro_score < 50:
            recommendations.append("• **Alerte sectorielle** : Secteur sensible aux conditions monétaires actuelles")
        elif sector in ['Energy', 'Materials'] and analysis.macro_score > 60:
            recommendations.append("• **Opportunité sectorielle** : Secteur favorisé par l'environnement inflationniste")
        
        return "\n".join(recommendations)
    
    def _format_market_cap(self, market_cap) -> str:
        """Formate la capitalisation boursière"""
        if not market_cap or market_cap == 'N/A':
            return 'N/A'
        
        try:
            cap = float(market_cap)
            if cap >= 1e12:
                return f"{cap/1e12:.1f}T $"
            elif cap >= 1e9:
                return f"{cap/1e9:.1f}B $"
            elif cap >= 1e6:
                return f"{cap/1e6:.1f}M $"
            else:
                return f"{cap:,.0f} $"
        except:
            return str(market_cap)