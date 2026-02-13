#!/bin/bash
echo "🔍 Test de propagation DNS pour helixone.fr"
echo "==========================================="
echo ""
echo "1️⃣ Enregistrements A :"
dig +short helixone.fr A
echo ""
echo "2️⃣ Test HTTP :"
curl -s -o /dev/null -w "Status: %{http_code}\n" https://helixone.fr/api/version.json 2>/dev/null || echo "❌ Pas encore accessible"
echo ""
echo "Si tu vois 4 adresses IP et Status: 200, c'est bon ! ✅"
