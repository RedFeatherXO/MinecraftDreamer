#!/bin/bash
# stop_all_minecraft.sh - Stoppt alle Minecraft Clients

# Lade Config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo -e "${RED}🛑 Stoppe alle Minecraft Clients...${NC}"

STOPPED=0
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    PID_FILE="$MINECRAFT_DIR/client_${i}.pid"
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        
        if ps -p $PID > /dev/null 2>&1; then
            echo "   Stoppe Client $i (PID $PID)..."
            kill $PID
            STOPPED=$((STOPPED + 1))
        fi
        
        rm "$PID_FILE"
    fi
done

# Fallback
pkill -f "launchClient" 2>/dev/null
pkill -f "gradle.*minecraft" 2>/dev/null

echo -e "${GREEN}✅ $STOPPED Clients gestoppt${NC}"

# Optional: Logs aufräumen
echo ""
read -p "Log-Dateien löschen? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$MINECRAFT_DIR"/client_*.log
    echo -e "${GREEN}✅ Logs gelöscht${NC}"
fi