#!/bin/bash
# view_logs.sh - Zeigt alle Client Logs live

# Lade Config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "📺 Client Logs anzeigen"
echo "======================"

# Check ob Log Files existieren
LOG_COUNT=$(ls "$MINECRAFT_DIR"/client_*.log 2>/dev/null | wc -l)

if [ "$LOG_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ Keine Log-Dateien gefunden!${NC}"
    echo "   Wurden die Clients gestartet?"
    echo "   Erwarteter Pfad: $MINECRAFT_DIR/client_*.log"
    exit 1
fi

echo "Gefundene Logs:"
for logfile in "$MINECRAFT_DIR"/client_*.log; do
    if [ -f "$logfile" ]; then
        SIZE=$(du -h "$logfile" | cut -f1)
        echo "   📄 $(basename "$logfile") ($SIZE)"
    fi
done

echo ""
echo "Optionen:"
echo "  1) Alle Logs gleichzeitig (tail -f)"
echo "  2) Einzelnen Log wählen"
echo "  3) Nur Errors anzeigen"
read -p "Wahl (1-3): " choice

case $choice in
    1)
        echo "📺 Zeige alle Logs (Ctrl+C zum Beenden):"
        tail -f "$MINECRAFT_DIR"/client_*.log
        ;;
    2)
        echo "Welchen Client? (0-$((NUM_CLIENTS-1)))"
        read -p "Client Nummer: " client_num
        LOG_FILE="$MINECRAFT_DIR/client_${client_num}_port_$((BASE_PORT + client_num)).log"
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "❌ Log nicht gefunden: $LOG_FILE"
        fi
        ;;
    3)
        echo "🔍 Zeige nur Errors:"
        grep -i "error\|exception\|failed" "$MINECRAFT_DIR"/client_*.log
        ;;
esac