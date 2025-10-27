"""
Service d'envoi d'emails avec SendGrid
"""

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from app.core.config import settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EmailService:
    """Service pour envoyer des emails via SendGrid"""

    def __init__(self):
        self.from_email = settings.FROM_EMAIL
        self.sendgrid_api_key = settings.SENDGRID_API_KEY
        self.app_name = settings.APP_NAME

    def _send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Envoie un email via SendGrid

        Args:
            to_email: Email du destinataire
            subject: Sujet de l'email
            html_content: Contenu HTML
            text_content: Contenu texte (optionnel)

        Returns:
            True si envoyé, False sinon
        """
        # Si pas de clé API configurée, log et skip
        if not self.sendgrid_api_key or self.sendgrid_api_key == "":
            logger.warning(f"[EMAIL SKIP] Pas de clé SendGrid configurée. Email non envoyé à {to_email}")
            logger.info(f"[EMAIL CONTENT] Sujet: {subject}")
            logger.info(f"[EMAIL CONTENT] Corps: {text_content or html_content[:200]}")
            return False

        try:
            message = Mail(
                from_email=Email(self.from_email, self.app_name),
                to_emails=To(to_email),
                subject=subject,
                html_content=Content("text/html", html_content)
            )

            if text_content:
                message.plain_text_content = Content("text/plain", text_content)

            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)

            if response.status_code in [200, 201, 202]:
                logger.info(f"✅ Email envoyé à {to_email}: {subject}")
                return True
            else:
                logger.error(f"❌ Erreur envoi email à {to_email}: Status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Exception lors de l'envoi email à {to_email}: {str(e)}")
            return False

    def send_welcome_email(self, to_email: str, first_name: Optional[str] = None) -> bool:
        """
        Envoie un email de bienvenue

        Args:
            to_email: Email du nouvel utilisateur
            first_name: Prénom de l'utilisateur

        Returns:
            True si envoyé
        """
        name = first_name if first_name else "Nouvel utilisateur"

        subject = f"Bienvenue sur {self.app_name} ! 🚀"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #00BFFF 0%, #0090FF 100%);
                   color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
        .button {{ display: inline-block; padding: 12px 30px; background: #00BFFF;
                  color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.app_name}</h1>
            <p>Analyse d'actions avec IA</p>
        </div>
        <div class="content">
            <h2>Bienvenue, {name} ! 🎉</h2>
            <p>Merci d'avoir créé votre compte sur <strong>{self.app_name}</strong>.</p>

            <p>Votre compte a été créé avec succès et vous disposez maintenant d'une <strong>licence d'essai de 14 jours</strong> pour découvrir toutes nos fonctionnalités :</p>

            <ul>
                <li>✅ Analyse technique avancée</li>
                <li>✅ Analyse fondamentale complète</li>
                <li>✅ Analyse de sentiment du marché</li>
                <li>✅ Évaluation des risques</li>
                <li>✅ Indicateurs macroéconomiques</li>
            </ul>

            <p>Commencez dès maintenant à analyser vos actions préférées et prenez des décisions éclairées !</p>

            <center>
                <a href="{settings.BACKEND_URL}" class="button">Commencer l'analyse</a>
            </center>

            <p style="margin-top: 30px; color: #666; font-size: 14px;">
                <strong>Besoin d'aide ?</strong><br>
                Notre équipe est là pour vous accompagner. N'hésitez pas à nous contacter.
            </p>
        </div>
        <div class="footer">
            <p>{self.app_name} - Votre assistant d'analyse boursière</p>
            <p>Cet email a été envoyé à {to_email}</p>
        </div>
    </div>
</body>
</html>
        """

        text_content = f"""
Bienvenue sur {self.app_name} !

Bonjour {name},

Merci d'avoir créé votre compte. Vous disposez maintenant d'une licence d'essai de 14 jours.

Fonctionnalités incluses :
- Analyse technique avancée
- Analyse fondamentale complète
- Analyse de sentiment
- Évaluation des risques
- Indicateurs macroéconomiques

Connectez-vous dès maintenant pour commencer : {settings.BACKEND_URL}

L'équipe {self.app_name}
        """

        return self._send_email(to_email, subject, html_content, text_content)

    def send_password_reset_email(self, to_email: str, reset_code: str) -> bool:
        """
        Envoie un email de réinitialisation de mot de passe

        Args:
            to_email: Email de l'utilisateur
            reset_code: Code de réinitialisation à 6 chiffres

        Returns:
            True si envoyé
        """
        subject = f"Réinitialisation de votre mot de passe {self.app_name}"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #00BFFF 0%, #0090FF 100%);
                   color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
        .code-box {{ background: white; border: 2px dashed #00BFFF; padding: 20px;
                     text-align: center; border-radius: 5px; margin: 20px 0; }}
        .code {{ font-size: 32px; font-weight: bold; color: #00BFFF;
                letter-spacing: 5px; font-family: monospace; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107;
                   padding: 15px; margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Réinitialisation du mot de passe</h1>
        </div>
        <div class="content">
            <h2>Demande de réinitialisation</h2>
            <p>Vous avez demandé à réinitialiser votre mot de passe pour votre compte <strong>{self.app_name}</strong>.</p>

            <p>Voici votre code de réinitialisation :</p>

            <div class="code-box">
                <div class="code">{reset_code}</div>
                <p style="margin-top: 10px; color: #666; font-size: 14px;">
                    Code valide pendant 1 heure
                </p>
            </div>

            <p>Pour réinitialiser votre mot de passe :</p>
            <ol>
                <li>Retournez sur l'application</li>
                <li>Entrez ce code dans le formulaire</li>
                <li>Choisissez votre nouveau mot de passe</li>
            </ol>

            <div class="warning">
                <strong>⚠️ Important :</strong><br>
                Si vous n'avez pas demandé cette réinitialisation, ignorez cet email.
                Votre mot de passe actuel reste inchangé.
            </div>

            <p style="margin-top: 30px; color: #666; font-size: 14px;">
                Ce code expire dans 1 heure et ne peut être utilisé qu'une seule fois.
            </p>
        </div>
        <div class="footer">
            <p>{self.app_name} - Analyse boursière avec IA</p>
            <p>Cet email a été envoyé à {to_email}</p>
        </div>
    </div>
</body>
</html>
        """

        text_content = f"""
Réinitialisation de votre mot de passe {self.app_name}

Vous avez demandé à réinitialiser votre mot de passe.

Votre code de réinitialisation : {reset_code}

Ce code est valide pendant 1 heure.

Pour réinitialiser votre mot de passe :
1. Retournez sur l'application
2. Entrez ce code dans le formulaire
3. Choisissez votre nouveau mot de passe

Si vous n'avez pas demandé cette réinitialisation, ignorez cet email.

L'équipe {self.app_name}
        """

        return self._send_email(to_email, subject, html_content, text_content)


# Instance globale du service
email_service = EmailService()
