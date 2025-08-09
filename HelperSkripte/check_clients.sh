#!/bin/bash
# check_clients.sh - Checkt ob alle Clients laufen

# Lade Config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "📊 Minecraft Client Status:"
echo "=========================="

# Suche PIDs im Minecraft Verzeichnis
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    PID_FILE="$MINECRAFT_DIR/client_${i}.pid"
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Client $i (PID $PID) läuft${NC}"
            
            # Zeige Port
            PORT=$((BASE_PORT + i))
            if netstat -tuln 2>/dev/null | grep -q ":$PORT " || ss -tuln 2>/dev/null | grep -q ":$PORT "; then
                echo -e "   📡 Port $PORT ist offen"
            else
                echo -e "   ${YELLOW}⚠️  Port $PORT noch nicht bereit${NC}"
            fi
            
            # Zeige letzte Log-Zeile
            LOG_FILE="$MINECRAFT_DIR/client_${i}_port_${PORT}.log"
            if [ -f "$LOG_FILE" ]; then
                LAST_LINE=$(tail -n 1 "$LOG_FILE" 2>/dev/null | head -c 60)
                echo "   Log: $LAST_LINE..."
            fi
        else
            echo -e "${RED}❌ Client $i (PID $PID) läuft nicht mehr${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Client $i nicht gestartet (keine PID-Datei)${NC}"
    fi
done

# Zeige alle Minecraft Java Prozesse
echo ""
echo "📋 Alle Minecraft Prozesse:"
ps aux | grep -E "minecraft|launchClient" | grep -v grep | head -5